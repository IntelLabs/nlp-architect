""" 
A module for handling path-pattern related processing, e.g. extracting OT->-AT patterns from MRP graphs. 
"""
from typing import List, Dict, Tuple, Any, Union
from collections import defaultdict, OrderedDict
import csv 
import json
import os
from pathlib import Path
from itertools import permutations, product
import numpy as np
from tqdm import tqdm
from graph import Graph as MrpGraph
import networkx as nx
import torch
from torch import nn

Span = Tuple[int, int]

MAX_HOPS_THRESHOLD = 2      # filtering criteria for path pattern auxilairy task data
NO_PREDICTION = nn.CrossEntropyLoss().ignore_index

"""
Utilities for data preprocessing w.r.t the path-pattern auxliliary tasks.
"""

def bio_to_spans(bio_labels: List[str]) -> Dict[str, List[Span]]:
    """ Return the spans given by BIO scheme, as {label: [(begin, end), ...]} """
    spans = defaultdict(list)
    span_label = None
    cur_span_info = None
    for token_id, bio_tag in enumerate(bio_labels):
        if (bio_tag == "O" or bio_tag.startswith("B")) and cur_span_info: 
            # finalize previous span (if any)
            cur_span_info["to"] = token_id  # span is exclusive
            spans[span_label].append((cur_span_info["from"], cur_span_info["to"]))
            cur_span_info = None
        if bio_tag.startswith("B"): # start new span 
            # start new span 
            span_label = bio_tag.split("-")[1]
            cur_span_info = {"from": token_id, "label": span_label}
    return spans

def to_nx(g: MrpGraph) -> nx.DiGraph:
    """ convert mrp_graph to networkx graph - e.g. for finding shortest paths """
    nxg = nx.DiGraph()
    # add nodes
    for node in g.nodes:
        nxg.add_node(node.id)
    # add edges
    for edge in g.edges:
        nxg.add_edge(edge.src, edge.tgt, label=edge.lab)
    return nxg 

def verify_0_index(mrp_graph):
    # modify mrp_graph (if needed) to verify nodes are 0-indexed (i.e. g.nodes[n.id] == n | n in g.nodes)  
    first_node_id = min(n.id for n in mrp_graph.nodes)
    for node in mrp_graph.nodes:
        node.id -= first_node_id
    for edge in mrp_graph.edges:
        edge.src -= first_node_id 
        edge.tgt -= first_node_id

def path_to_pattern(nxg: nx.DiGraph, path: List[int]) -> str:
    """ 
    Return the path-pettern of the path in the graph. 
    Pattern specifies the edge labels on the path and their direction. 
    E.g. for a graph X <--a1-- Z --a2 --> Y (a1 and a2 being the relation labels), 
    there is a path from X to Y (X~>~Y), and its pattern would be '[X, a1^-1, Z, a2, Y]'
    """
    # Comprise the path pattern from representation of nodes and edges along the path
    def node_repr(node_id: int) -> str: 
        """ don't specify anything about nodes """
        return "*"
    def edge_repr(u: int, v: int) -> str:
        """ return dependency-label. Assuming (u,v) in nxg.edges. """
        return nxg.edges[(u,v)]['label']
    # create and fill the pattern as a list of node/edge representations
    pattern = ["OT"] # first node
    # iteratively add <edge>, <node> from path
    for i in range(1, len(path)):
        # represent edge - edge might be in reverse direction
        hop = (path[i-1], path[i]) # the next hop in path is from path[i-1] to path[i]
        assert hop in nxg.edges or tuple(reversed(hop)) in nxg.edges, "nx_path contain hops not in DiGraph"
        if hop in nxg.edges: # hop is in standard direction of arc
            pattern.append(edge_repr(*hop))
        else:   # hop is in reverse direction of arc
            pattern.append(edge_repr(*reversed(hop)) + "^-1") # concatenate reverse-dir marker 
        # represent node
        pattern.append(node_repr(path[i])) 
    pattern[-1] = "ASP" # override last node in pattern with "ASP" just for pattern readability 
    return json.dumps(pattern)
 
def get_OT2AT_patterns(mrp_graph, bio_labels):
    """
    Retrieve the path pattern from OT tokens to related AT token. 
    Returns a per-token list, being the class labels for the linguistically-informed auxiliary task.
    """
    # augment ABSA data (i.e aspect and opinion terms) into mrp_graph
    absa_spans = bio_to_spans(bio_labels)
    for n in mrp_graph.nodes:   # init with False
        n.aspect_term = False
        n.opinion_term = False
    for aspect_span in absa_spans["ASP"]:   # Then fix by absa_spans
        for idx in range(*aspect_span):
            mrp_graph.nodes[idx].aspect_term = True
    for opinion_span in absa_spans["OP"]:  
        for idx in range(*opinion_span):
            mrp_graph.nodes[idx].opinion_term = True
    mrp_graph.aspect_spans = absa_spans["ASP"]
    mrp_graph.opinion_spans = absa_spans["OP"]

    verify_0_index(mrp_graph)
    
    nxg = to_nx(mrp_graph) 
    nxug = nxg.to_undirected()
    
    # enumerate all possible OT-AT shortest paths (possible combinatorically)
    def filtered(lst): return [e for e in lst if e is not None]
    def shortest_path(source, target):
        """ source and target are token indices. Return None if no path."""
        try:
            return nx.shortest_path(nxug, source=source, target=target)
        except nx.NetworkXNoPath:
            return None
    def get_nx_undirected_path(opinion: Span, aspect: Span) -> Union[List[int], None]:
        """ 
        Returns the list of nodes on the undirected (shortest) path from opinion to aspect.
        In case opinion or aspect spans have multiple words, take the shortest path among all
        paths - it would probably be between the head tokens of each term.   
        """
        all_paths = [shortest_path(source_tokid, target_tokid) 
                    for source_tokid in range(*opinion)
                    for target_tokid in range(*aspect)]
        all_paths = filtered(all_paths) # remove None instances
        if not all_paths:
            return None # no path at all
        shortest_AT2OT_path = sorted(all_paths, key=len)[0] # return shortest
        return shortest_AT2OT_path
     
    all_possible_pairs = list(product(absa_spans["OP"], absa_spans["ASP"]))
    all_paths = defaultdict(list) 
    for op_span, asp_span in all_possible_pairs:
        path = get_nx_undirected_path(op_span, asp_span) # path is list of node-ids (=token indices)
        if path and len(path)-1 <= MAX_HOPS_THRESHOLD:   # len(path) is #-nodes, it-1 is #-edges 
            # further filtering criteria on taken paths - path length (number of hops i.e. edges) 
            all_paths[op_span].append(path) 
                
    # all_paths includes a path for any (OT,AT) pair (in case there is one).
    # Thus, it includes multiple paths for a single OT in cases where there are multiple ATs.
    # We should take the shortest for each OT
    selected_paths = [(op_span, sorted(paths, key=len)[0])
                      for op_span, paths in all_paths.items() if paths]
    # translate path into path-pattern - the class-label for predict
    # selected_patterns is a list of (path, source-op-token, destination-at-tokem, pattern-repr)
    selected_patterns = [(path, 
                          path[0],  # first node in path is of OT
                          path[-1], # last node in path is of AT
                          path_to_pattern(nxg, path))
                         for op_span, path in selected_paths]
    """
    Auxiliary task would be a token classification problem focused on OTs.
    Classification scheme -  
        Opinion tokens:
        For other OT tokens inside the same OT span, "pattern label" would be "OT-Secondary";  
        For tokens inside OT spans with no path, "pattern label" would be "OT-W\O-Path";
        For the OT token starting the path, "pattern label" would be the pattern;
        Aspect tokens:
        For AT tokens in end of path - "AT-Head"; 
        For other AT tokens in these AT spans - "AT-Secondary";
        For tokens inside AT-spans that aren't included in selected paths, "pattern label" would be "AT-W\O-Path";
        For other, non-term tokens - "O"
    """
    # AT tokens that occur (as targets) in selected paths
    selected_aspect_tokens = {dst_asp_token for _,_,dst_asp_token,_ in selected_patterns} 
    selected_aspect_spans = {span for span in absa_spans["ASP"] if any(i in range(*span) for i in selected_aspect_tokens)}
    all_opinion_tokens = {i for span in absa_spans["OP"] for i in range(*span)}
    all_aspect_tokens = {i for span in absa_spans["ASP"] for i in range(*span)}
    op_span_in_path_tokens = {i for span,path in selected_paths for i in range(*span)}
    asp_span_in_path_tokens = {i for span in selected_aspect_spans for i in range(*span)}
    
    pattern_labels = ["O"] * len(bio_labels)
    # set labels for AT and OT tokens by classes
    def set_pattern_label(indices: List[int], label: str):
        for i in indices:
            pattern_labels[i] = label
    set_pattern_label(all_opinion_tokens, "OT-W\\O-Path")
    set_pattern_label(all_aspect_tokens, "AT-W\\O-Path")
    # override more generic classes
    set_pattern_label(op_span_in_path_tokens, "OT-Secondary")
    set_pattern_label(asp_span_in_path_tokens, "AT-Secondary")
    # override again
    set_pattern_label(selected_aspect_tokens, "AT-Head")
   
    # The Major task: set labels for selected OT tokens 
    # The label would be (destination-AT-token, the OT->-AT pattern repr)
    for path, src_op_token, dst_asp_token, pattern in selected_patterns:
        # encode (AT-idx, pattern) tuple with '###' sep
        pattern_labels[src_op_token] = f"{dst_asp_token}###{pattern}"
    
    return pattern_labels
  
  
"""
Modules for predicting the path-pattern related auxiliary tasks. 
"""
class AspectMatcher(nn.Module):
    """ 
    A Module for predicting the Aspect Terms corresponding to given Opinion Terms.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # add classifiers for auxliary tasks
        self.asp_matcher = nn.Linear(config.hidden_size, config.hidden_size)
        """ Also consider 2-layer MLP:
        self.asp_matcher = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(config.hidden_size, config.hidden_size)),
            ('activation', nn.ReLU()),
            ('lin2', nn.Linear(config.hidden_size, config.hidden_size))
            ]))
        """

    def forward(
        self,
        sequence_output,
        opinion_mask,
        tgt_asp_indices
    ):
        output_dict = {}
        loss_fct = nn.CrossEntropyLoss()
        # define sizes for readability - B=Batch size, S=Sequence length, H=Hidden size
        B,S,H = sequence_output.shape
        # iterate targets (i.e. opinion terms) and stack them in order to jointly-compute & sum their losses
        asp_logits_stack, target_asp_idx_stack = [], []
        for batch_idx, op_idx in zip(*opinion_mask.nonzero(as_tuple=True)):
            target_asp_idx = tgt_asp_indices[batch_idx, op_idx]
            # the score (logit) for each token j (given the OT token i) is
            # computed by Xi*W*Xj, where W is a HxH parameter matrix W (in `self.asp_matcher`).
            op_vec = sequence_output[batch_idx, op_idx] # op_vec == Xi   (H,)
            query_vec = self.asp_matcher(op_vec)        # query_vec == Xi*W   (H,)
            batch_seq_output = sequence_output[batch_idx]  #  == X1..Xs, i.e. (S, H)
            query_mat = query_vec.view(1, H).repeat(S, 1)  #  for dot-producting location-wise. query_mat.size is (S, H)
            asp_logits = torch.bmm(query_mat.view(S, 1, H), batch_seq_output.view(S, H, 1)).squeeze()
            asp_logits_stack.append(asp_logits)
            target_asp_idx_stack.append(target_asp_idx)
        if asp_logits_stack: 
            asp_logits_stack = torch.stack(asp_logits_stack)
            target_asp_idx_stack = torch.stack(target_asp_idx_stack)
            # compute aux-loss, add it to outputs
            asp_match_aux_loss = loss_fct(asp_logits_stack, target_asp_idx_stack)
            output_dict.update(asp_match_aux_loss=asp_match_aux_loss, 
                            asp_match_aux_logits=asp_logits_stack, 
                            asp_match_gold_labels=target_asp_idx_stack)
        return output_dict


class PatternPredictor(nn.Module):
    """ 
    A Module for predicting the path-pattern from given Opinion Terms to Aspect Terms.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.classifier = nn.Linear(config.hidden_size, len(config.all_patterns))

    def forward(
        self,
        sequence_output,
        opinion_mask,
        patterns
    ):
        output_dict = {}
        loss_fct = nn.CrossEntropyLoss()
        
        patt_aux_logits = self.classifier(sequence_output)
        num_pattern_classes = len(self.config.all_patterns)
        # Only keep active parts of the loss, by opinion-term mask (masking out non-opinion-term tokens)
        aux_active_loss = opinion_mask.view(-1) == 1
        aux_active_logits = patt_aux_logits.view(-1, num_pattern_classes)
        aux_active_labels = torch.where(
            aux_active_loss, patterns.view(-1),
            torch.tensor(NO_PREDICTION).type_as(patterns)
        )
        # compute aux-loss, add it to outputs
        patt_aux_loss = loss_fct(aux_active_logits, aux_active_labels)
        output_dict.update(patt_aux_loss=patt_aux_loss, 
                           patt_aux_logits=patt_aux_logits,
                           patt_aux_labels=aux_active_labels)
        return output_dict

"""
Evaluation methods
"""
def classification_accuracy(gold_list, pred_list) -> float: 
    assert len(pred_list) == len(gold_list), "num of predictions not aligned with num of gold labels"
    n_correct = sum([1 for pred, gold in zip(pred_list, gold_list) if pred==gold])
    return n_correct / float(len(gold_list))
        
def asp_match_task_eval(outputs) -> OrderedDict:
    "Evaluation for Matched-Aspect detection auxiliary task. Called for both Val and Test."
    predictions = [x["asp_match_aux_preds"] for x in outputs if "asp_match_aux_preds" in x]
    targets = [x["asp_match_aux_target"] for x in outputs if "asp_match_aux_target" in x]
    # stack all instances (i.e. OP tokens with path)
    predictions = np.concatenate(predictions, axis=0)
    predictions = np.argmax(predictions, axis=-1)
    targets = np.concatenate(targets, axis=0)
    assert len(predictions) == len(targets), "num of predictions not aligned with num of gold labels"

    calc = lambda f: torch.tensor(f(targets, predictions))
    results = OrderedDict({
        f"asp-match_accuracy": calc(classification_accuracy)
    })
    return results

def pattern_task_eval(outputs) -> OrderedDict:
    "Evaluation for Path-Pattern Classification auxiliary task. Called for both Val and Test."
    predictions = [x["patt_aux_preds"] for x in outputs if "patt_aux_preds" in x]
    targets = [x["patt_aux_target"] for x in outputs if "patt_aux_target" in x]
    # stack all instances (i.e. OP tokens with path)
    predictions = np.concatenate(predictions, axis=0)
    predictions = np.argmax(predictions, axis=-1)
    targets = np.concatenate(targets, axis=0)
    assert len(predictions) == len(targets), "num of predictions not aligned with num of gold labels"

    # count correct predictions out of meaningful predictions (i.e. predictions for OT token)
    n_true, n_total = 0, 0
    for pred, gold in zip(predictions.reshape(-1), targets.reshape(-1)):
        if gold != NO_PREDICTION:
            n_total += 1
            if gold == pred:
                n_true += 1
    # report accuracy
    accuracy = float(n_true) / n_total
    results = OrderedDict({
        f"pattern_accuracy": torch.tensor(accuracy)
    })
    return results