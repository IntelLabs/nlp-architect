#%%
import csv 
import json
import os
import argparse
from pathlib import Path
from itertools import permutations
from transformers import BertTokenizer
from tqdm import tqdm
import spacy
from spacy.tokens import Doc
from graph import Graph as MrpGraph


LIBERT_DIR = Path(os.path.realpath(__file__)).parent
DATA_DIR = LIBERT_DIR / 'data'
CONLL_DIR = DATA_DIR / 'conll'
DOMAINS_DIR = CONLL_DIR / 'domains_all'

default_ud_model = "en_ud_model_lg" 

class SpacyWithBertTokenizer:
    def __init__(self, spacy_model_name):
        self.parse_cache = {}
        self.nlp = self.get_spacy_with_bert_tokenizer(spacy_model_name)
        #  For UD dependency parses, first download a SpaCy UD model- run:
        # > pip install https://storage.googleapis.com/en_ud_model/en_ud_model_lg-1.1.0.tar.gz
        #  Then, replace default `spacy_model` with "en_ud_model_lg" - replace the last line with:
        # self.nlp = self.get_spacy_with_bert_tokenizer(spacy_model="en_ud_model_lg")
        
    @staticmethod
    def get_spacy_with_bert_tokenizer(spacy_model):
        nlp = spacy.load(spacy_model, disable=["ner", "vectors", "textcat"])
        bert_tokenizer_m = ModifiedBertTokenizer.from_pretrained('bert-base-uncased')
        nlp.tokenizer = Tokenizer(nlp.vocab, bert_tokenizer_m)
        return nlp

    def pipe_batch(self, batch):
        cached_idx = {}
        not_in_cache_idx = {}
        not_in_cache = []
        for i, text in enumerate(batch):
            if text in self.parse_cache:
                cached_idx[i] = self.parse_cache[text]
            else:
                not_in_cache_idx[i] = len(not_in_cache)
                not_in_cache.append(text)

        parsed_docs = list(self.nlp.pipe(not_in_cache))
        res = []
        for i in range(len(batch)):
            if i in cached_idx:
                res.append(cached_idx[i])
            else:
                parsed_doc = parsed_docs[not_in_cache_idx[i]]
                self.parse_cache[batch[i]] = parsed_doc
                res.append(parsed_doc)
        return res

    def pipe(self, word_lists):
        docs = []
        for s_batch in tqdm(batcher([' '.join(words) for words in word_lists], n=20)):
            docs.extend(self.pipe_batch(s_batch))

        res_list = []
        for doc in docs:
            sub_toks = [' '.join(sub_tok) if len(sub_tok) > 1 else '' for sub_tok in doc.user_data]
            toks = [t.text for t in doc]
            pos = [t.tag_ for t in doc]
            heads = [t.head.i for t in doc]
            head_words = [t.head.text for t in doc]
            rels = [t.dep_ for t in doc]
            assert len(toks) == len(sub_toks) == len(pos) == len(heads) == len(rels)
            res_list.append((toks, pos, sub_toks, heads, head_words, rels))
        return res_list

def batcher(iterable, n=1):
    iter_len = len(iterable)
    for ndx in range(0, iter_len, n):
        yield iterable[ndx: min(ndx + n, iter_len)]

def conll_iter(f):
    toks, labels = [], []
    for line in f:
        line = line.strip()
        if not line:
            yield tuple(toks), tuple(labels)
            toks, labels = [], []
        else:
            split = line.split('\t')
            labels.append(split[1])
            toks.append(split[0])

def parse_file(txt_path, spacy_bert_tok, subdir, overwrite=True):
    conll_path_parts = Path(txt_path).parts
    out_parent_dir = DATA_DIR / 'csv' / subdir / conll_path_parts[-2] 
    out_path = out_parent_dir / conll_path_parts[-1].replace('.txt','.csv')
    os.makedirs(out_parent_dir, exist_ok=True)

    if overwrite or not os.path.exists(out_path):
        with open(txt_path, encoding='utf-8') as input_f:
            space_tok_reviews = []
            reviews_tok_labels = []
            for toks, labels in conll_iter(input_f):
                space_tok_reviews.append(toks)
                reviews_tok_labels.append(labels)

            with open(out_path, 'w', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                parsed_reviews = spacy_bert_tok.pipe(space_tok_reviews)
                writer.writerow(['TOKEN', 'LABEL', 'HEAD', 'HEAD_WORD', 'DEP_REL', 'POS', 'SUB_TOKENS'])

                for word_list, parsed_review, labels in \
                    zip(space_tok_reviews, parsed_reviews, reviews_tok_labels):
                    toks = parsed_review[0]
                    assert len(toks) == len(labels)
                    assert tuple(toks) == word_list
                    for tok, pos_tag, sub_tok, head, head_word, rel, label in zip(*parsed_review, labels):
                        writer.writerow([tok, label, head, head_word, rel, pos_tag, sub_tok])
                    writer.writerow(['_'] * 7)
                    writer.writerow(['_'] * 7)

class ModifiedBertTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        # kwargs['do_lower_case'] = False
        super(ModifiedBertTokenizer, self).__init__(*args, **kwargs)

    def tokenize(self, text, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.
        """
        def split_on_tokens(tok_list, text, res_toks=None):
            if not text:
                return []
            if not tok_list:
                tokenized = self._tokenize(text, **kwargs)
                return tokenized
            tok = tok_list[0]
            split_text = text.split(tok)
            if res_toks is not None:
                res_toks.extend(split_text)
            sub_tokens = sum((split_on_tokens(tok_list[1:], sub_text.strip()) + [tok] \
                        for sub_text in split_text), [])[:-1]
            return sub_tokens

        added_tokens = list(self.added_tokens_encoder.keys()) + self.all_special_tokens
        tokens = []
        sub_tokens = split_on_tokens(added_tokens, text, tokens)
        return tokens, sub_tokens

class Tokenizer:
    """ 
    A Tokenizer for Spacy pipeline, that wraps a BERT tokenizer 
    and keeps sub-token information under `doc.user_data`.
    
    doc.user_data would be a list of token-info, where each token-info 
    is itself a non-empty list of subtokens.
    len(doc) == len(doc.user_data) always applies.  
    """  
    def __init__(self, vocab, bert_tokenizer):
        self.vocab = vocab
        self.bert_tokenizer = bert_tokenizer

    def __call__(self, text):
        res = []
        words = text.split()
        sub_tokens = []
        for word in words:
            bert_token, sub_tok = self.bert_tokenizer.tokenize(word)
            sub_tokens.append(sub_tok)
            res.extend(bert_token)
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        doc = Doc(vocab=self.vocab, words=res, spaces=spaces)
        doc.user_data = sub_tokens
        return doc

def parse_cross_domain(domains: list, spacy_model: str, subdir: str, splits=3, modes=('train', 'dev', 'test'), overwrite=True):
    spacy_bert_tok = SpacyWithBertTokenizer(spacy_model_name=spacy_model)

    for domain_a, domain_b in permutations(domains, r=2):
        print('Setting: ' + domain_a + ' to ' + domain_b)
        for split in range(splits):
            for ds in tqdm(modes):
                ds_file = str(CONLL_DIR / f'{domain_a}_to_{domain_b}_{split + 1}' / f'{ds}.txt')
                print(ds_file)
                parse_file(ds_file, spacy_bert_tok, subdir=subdir, overwrite=overwrite)

def parse_in_domain(domains: list, spacy_model: str, subdir: str):
    spacy_bert_tok = SpacyWithBertTokenizer(spacy_model_name=spacy_model)

    for domain in tqdm(domains):
        ds_file = str(DOMAINS_DIR / f'{domain}.txt')
        print('File: ' + ds_file)
        parse_file(ds_file, spacy_bert_tok, subdir=subdir)

def get_dep_relations_from_csv(subdir: str):
    """ Retrieve all dependecny relation labels from ALL CSVs """
    import pandas as pd
    import glob
    path = DATA_DIR / 'csv' / subdir
    all_files = glob.glob(str(path) + "/*/*.csv")
    dfs = [pd.read_csv(filename, index_col=None)
        for filename in all_files]
    df = pd.concat(dfs, axis=0, ignore_index=False)
    # dep_relation from dataframe contains contenated dependency relations
    #  for graph-structured formalisms, so flatten it 
    dep_relations = {rel.lower() 
                     for line in set(df['DEP_REL']) 
                     for rel in line.strip().split('~')} 
    # for no-heads, dep_relations will include a "_" relation label, representing "no head"
    return list(sorted(dep_relations))

# load global bert tokenizer - for efficiency
bert_tokenizer = ModifiedBertTokenizer.from_pretrained('bert-base-uncased') 

def graph_as_head_dependent_relations(graph: MrpGraph, edge_direction="head~>dependent"):
    """
    Represent the graph as token-wise head-dependent relations.
    :Return: an exhaustive dict {node-id: <heads>} for every node-id in self.nodes.
    <heads> is a tuple of pairs (head-id, relation) of the nodes having an edge to current node-id.
    The tuple can be empty (for nodes with no incoming edges), can be 1-tuple (only one "head" pair),
    or k-tuple (for multi-headed tokens).
    For the `top` node, <heads> will be (("ROOT","ROOT"),)
    
    :arg edge_direction: What is the interpretation of edge direcitonality. Default
        interpretation ("head~>dependent") is that source node is considered head of destination node 
    """
    from collections import defaultdict
    res = defaultdict(tuple)
    # insert all edges info
    if edge_direction == "head~>dependent":
        # source node is considered head of destination node
        for edge in graph.edges:
            res[edge.tgt] += ((edge.src, edge.lab),)
    else:
        # destination node is considered head of source node
        for edge in graph.edges:
            res[edge.src] += ((edge.tgt, edge.lab),)
    # have mark top with "ROOT" as head
    for node in graph.nodes:
        if node.is_top:
            res[node.id] += (("ROOT", "ROOT"),)
    return res


def mrp_graph_to_parsed_review(mrp_graph, word_list):
    """ Return token-list of (tok, pos_tag, sub_tok, head, head_word, rel) """
    global bert_tokenizer
    mrp_graph = mrp_graph._full_sentence_recovery()
    assert len(mrp_graph.input.split()) == len(mrp_graph.nodes), "assuming mrp.input is tokenized by space but found mismatch with number of nodes."
    graph_heads = graph_as_head_dependent_relations(mrp_graph)     
    node_id2index = {n.id:i for i,n in enumerate(mrp_graph.nodes)}
    token_infos = []
    for node, word in zip(mrp_graph.nodes, word_list):
        node_surface = mrp_graph.input[node.anchors[0]['from']:node.anchors[0]['to']]
        assert node_surface == word, f"node surface is {node_surface} while token is {word}"
        tok, sub_toks = bert_tokenizer.tokenize(word)
        sub_tok_repr = ' '.join(sub_toks) if len(sub_toks) > 1 else ''
        tok = tok[0]
        assert word == tok, f"bert first-order tokenized tok ({tok}) differs from conll word ({word})"
        # Not all nodes are predicted with properties such as 'pos'; 
        # node.properties can be None
        if node.properties and 'pos' in node.properties:
            pos_tag = node.values[node.properties.index('pos')]
        else:
            pos_tag = '.' if node.label == 'non' else '_'
        """
        For representing bilexical graphs in our Conll-like CSVs, where a token might have 0 or >1 heads, 
        we use '_' to denote None heads and use '~' as an intra-cell delimiter for multiple entries.
        """
        if graph_heads[node.id]:
            head_node_ids, head_relations = zip(*graph_heads[node.id])
            # special handle of "ROOT" - the head-node-id is denoted by str "ROOT" instead of integer,
            # replace it with the node-id of the dependent token itself (Root is attending itself) 
            head_node_ids = [nid if nid is not "ROOT" else node.id
                             for nid in head_node_ids]
            head_idxs = [node_id2index[nid] for nid in head_node_ids]
            head_words = [word_list[idx] for idx in head_idxs]		
            # Represent heads, head_words and head_relations for CSV encoding.	
            head = '~'.join([str(idx) for idx in head_idxs])
            head_word = '~'.join([word for word in head_words])
            rel = '~'.join([dep_relation for dep_relation in head_relations])
        else:
            # handle headless tokens
            head = '_'
            head_word = '_'
            rel = '_'
        token_infos.append((tok, pos_tag, sub_tok_repr, head, head_word, rel))
    return token_infos         
    

def convert_mrp_file(conll_file: str, parse_mrp_file: str, subdir: str):
    """ Generate a CSV file with linguistic information based on an ABSA-conll file 
    with raw + BIO data, along with an MRP file with semantic parses. """
    out_path = DATA_DIR / "csv" / subdir / Path(conll_file).parent.name / (Path(conll_file).stem + ".csv")
    os.makedirs(Path(out_path).parent, exist_ok=True)

    # Read parses (mrp files)
    mrp_graphs = []
    with open(parse_mrp_file) as fin:
        for line in fin:
            mrp_dict = json.loads(line.strip())
            mrp_graphs.append(MrpGraph.decode(mrp_dict))
    # Read ABSA BIO labels (conll files)
    with open(conll_file, encoding='utf-8') as input_f:
        space_tok_reviews = []
        reviews_tok_labels = []
        for toks, labels in conll_iter(input_f):
            space_tok_reviews.append(toks)
            reviews_tok_labels.append(labels)

    with open(out_path, 'w', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['TOKEN', 'LABEL', 'HEAD', 'HEAD_WORD', 'DEP_REL', 'POS', 'SUB_TOKENS'])
        for word_list, mrp_graph, labels in zip(space_tok_reviews, mrp_graphs, reviews_tok_labels):
            parsed_review = mrp_graph_to_parsed_review(mrp_graph, word_list)
            for (tok, pos_tag, sub_tok, head, head_word, rel), label in zip(parsed_review, labels):
                # print([tok, label, head, head_word, rel, pos_tag, sub_tok])
                writer.writerow([tok, label, head, head_word, rel, pos_tag, sub_tok])
            writer.writerow(['_'] * 7)
            writer.writerow(['_'] * 7)


def prepare_from_semantic_parses(formalism: str, domains, splits=3, modes=('train', 'dev', 'test')):
    # formalism = "dm"
    # splits=3
    # modes=('train', 'dev', 'test')
    PARSES_DIR = DATA_DIR / "semantic_parses" / formalism
    for domain_a, domain_b in permutations(domains, r=2):
        print('\nSetting: ' + domain_a + ' to ' + domain_b)
        for split in range(splits):
            for ds in tqdm(modes):
                conll_file = str(CONLL_DIR / f'{domain_a}_to_{domain_b}_{split + 1}' / f'{ds}.txt')
                parse_mrp_file = str(PARSES_DIR / f'{domain_a}_to_{domain_b}_{split + 1}' / f'{ds}.mrp')
                print(f"Converting {conll_file}...")
                convert_mrp_file(conll_file, parse_mrp_file, subdir=formalism)


#%%
if __name__ == "__main__":
    default_spacy_model = "en_core_web_lg"
    arg_parser = argparse.ArgumentParser(description="Prepare linguistic information for ABSA data")
    arg_parser.add_argument("method", default = "spacy", choices = ["spacy", "ud", "mrp"], type=str, 
                            help="Source of the linguistic information - "
                            "spacy model or pre-parsed MRP files in data/semantic_parses")
    arg_parser.add_argument("-m", "-f", "--model", "--formalism", default = None, 
                            type=str, dest='mod', metavar="MODEL or FORMALISM",
                            help="name of SpaCy model, or name of semantic formalism")
    args = arg_parser.parse_args()
    
    if args.method in ("spacy", "ud"):
        # Run spacy model to parse all absa data
        if args.method == "ud" and args.mod == None:
            args.mod = default_ud_model 
        elif args.method == "spacy" and args.mod == None:
            args.mod = default_spacy_model
        subdir = args.method    # sub-directory within "data/csv/"
        parse_cross_domain(domains=['restaurants', 'laptops', 'device'], spacy_model=args.mod, subdir=subdir)
        parse_in_domain(domains=['restaurants', 'laptops', 'device'], spacy_model=args.mod, subdir=subdir)
    elif args.method == "mrp":
        # Convert MRP-formatted files into csv
        subdir = formalism = args.mod.lower() 
        prepare_from_semantic_parses(domains=['restaurants', 'laptops', 'device'], 
                                     formalism=formalism)
    
    # Save label set to label.txt
    with open(DATA_DIR / 'labels.txt', 'w', encoding='utf-8') as labels_f:
        labels_f.write('\n'.join(['O', 'B-ASP', 'I-ASP', 'B-OP', 'I-OP']))

    # Prepare dep_relations.txt
    dep_relations = get_dep_relations_from_csv(subdir)
    with open(DATA_DIR / 'csv' / subdir / 'dep_relations.txt', 'w', encoding='utf-8') as dep_rel_labels_f:
        dep_rel_labels_f.write('\n'.join(dep_relations))


