# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION. 
# Copyright 2019-2020 Intel Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Cross-domain ABSA fine-tuning: utilities to work with SemEval-14/16 files. """

# pylint: disable=logging-fstring-interpolation, not-callable
import os
from dataclasses import dataclass
import csv
import shutil
from enum import Enum
from collections import defaultdict
from typing import Counter, List, Optional, Union, Tuple
from os.path import realpath
from argparse import Namespace
from pathlib import Path
from datetime import datetime as dt
import torch

from torch.nn import CrossEntropyLoss
from torch import tensor
import torch.nn.functional as F
from pytorch_lightning import _logger as log
from pytorch_lightning.core.saving import load_hparams_from_yaml
from transformers import PreTrainedTokenizer
from seqeval.metrics.sequence_labeling import get_entities
import numpy as np
import pandas as pd

from significance import significance_from_cfg as significance
import path_patterns

LIBERT_DIR = Path(realpath(__file__)).parent
LOG_ROOT = LIBERT_DIR / 'logs'

AUX_SEP = "###"     # internal seperator in the auxiliary label data column  


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """
    guid: str
    words: List[str]
    labels: Optional[List[str]]
    heads: Optional[List[List[int]]]
    head_words: Optional[List[List[str]]]
    pos_tags: Optional[List[str]]
    sub_toks: Optional[List[List[str]]]
    aux_task_labels: Optional[List[str]]

@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    dep_heads: Optional[List[float]] = None
    # auxiliary task data
    op2at_opinion_mask: Optional[List[int]] = None # masks-in indices to compute loss for on auxiliary task
    op2at_patterns: Optional[List[int]] = None # sequence of pattern label ids (length==max sequence)
    op2at_tgt_asp_index: Optional[List[int]] = None # sequecne of absolute index of target AT token (length==max sequence)
    
class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.csv")
    guid_index = 1
    examples = []
    empty_row = ['_'] * 7
    with open(file_path, encoding='utf-8') as f:
        words, labels, heads, head_words, pos_tags, sub_toks, aux_labels = [], [], [], [], [], [], []
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row == empty_row:
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels,
                                                    heads=heads, head_words=head_words, pos_tags=pos_tags, 
                                                    sub_toks=sub_toks, aux_task_labels=aux_labels))
                    guid_index += 1
                    words, labels, heads, head_words, pos_tags, sub_toks, aux_labels = [], [], [], [], [], [], []
            else:
                word, label, head, head_word, syn_rel, pos_tag, sub_tok, aux_label = row
                """
                For representing bilexical graphs in our Conll-like CSVs, 
                where a token might have 0 or >1 heads, 
                we use '_' to denote None heads, 
                and use '~' as an intra-cell delimiter for multiple entries
                in `head`, `head_word` and `syn_rel`.
                """
                words.append(word)
                labels.append(label)
                heads.append([int(h) for h in head.split('~')] if head is not "_" else [])
                head_words.append(head_word.split('~') if head_word is not "_" else [])
                pos_tags.append(pos_tag)
                sub_toks.append(sub_tok.split() if sub_tok else [word])
                aux_labels.append(aux_label)
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels,
                                            heads=heads, head_words=head_words,
                                            pos_tags=pos_tags, sub_toks=sub_toks, 
                                            aux_task_labels=aux_labels))
    return examples

def idxs_to_mask(indices, seq_len=64) -> tensor:
    """ 
    Get a tensor/array which it's last dimension stand for lists of indices,
    and return a tensor of shape (indices.shape(:-1) + (seq_len,)),
    where last dimension is now a binary mask - i.e. 1 in only specified indices 
    (considering 1-offset for special [CLS] token). 
    """
    if not indices:
        return torch.zeros(seq_len)
    if not isinstance(indices, F.Tensor): indices = tensor(indices)
    as_one_hot_matrix = F.one_hot(indices, seq_len-1)
    masks = as_one_hot_matrix.sum(dim=-2)
    # preppend zeros on last dimension of output mask, correspinding to [CLS] token
    def preppend_zeros(x: tensor):
        return torch.cat((torch.zeros(x.shape[:-1], dtype=x.dtype).unsqueeze(-1), x), -1)
    masks = preppend_zeros(masks)
    assert masks.shape == indices.shape[:-1] + (seq_len,), "masks shape error" 
    return masks

def mask_to_idxs(mask: tensor, seq_len=64) -> List[int]:
    pass #TODO

def pad_heads(a):
    target = np.zeros((64, 64), float)
    target[:a.shape[0], :a.shape[1]] = a[:64, :64]
    return target
    
def apply_heads_to_subtokens(heads, sub_tokens, zero_sub_tokens=False):
    for i in range(len(sub_tokens) - 1, -1, -1):
        for _ in range(1, len(sub_tokens[i])):
            if zero_sub_tokens:
                sub_token_row = [0 for _ in heads[i]]
            else:
                sub_token_row = heads[i].copy()
            heads.insert(i + 1, sub_token_row)

    for row in heads:
        # zero [CLS] (ROOT) column?
        for i in range(len(sub_tokens) - 1, -1, -1):
            for _ in range(1, len(sub_tokens[i])):
                row.insert(i + 2, 0)

    # insert zeros row for [CLS] token
    heads.insert(0, [0] * (len(heads) + 1))
    return np.array(heads)

def binarize(preds: List[int]):
    "take an sequence of token-idxs, and convert each token-idx to a one-hot vector - returns a matrix"
    res = []
    for pred in preds:
        res.append([1 if i == pred else 0 for i in range(len(preds) + 1)])
    return res

def binarize_multi_hot(heads: List[List[int]]):
    """
    While `binarize` take an sequence of token-idxs, and convert each token-idx to a one-hot vector 
    (at length of sequence + 1), this function is a variant that handles a sequence of List[token-idxs].
    The function is used for supporting multiple token-heads (for graph-structured, vs. tree-structured, 
    linguistic formalisms, e.g. semantic dependencies). Therefore, a token's head is represented in the input 
    (and in `heads`) as a list of integeres rather than a single integer. 
    In output matrix, each row will be a "multi-hot" vector.
     
    E.g. binarize_multi_hot([[1],[0,1]]) -> [[0,1,0], 
                                             [1,1,0]]
    Args:
        heads (List[List[int]]): a sequence-lengthed list of lists of head-indices. 

    Returns:
        [List[List[int]]]: a list of "multi-hot" binary vectors. 
    """
    res = []
    for head in heads:
        vec = [1 if i in head else 0 for i in range(len(heads) + 1)]
        # special handling for tokens with no-heads (in which head==[]):
        # Don't represent as all-zero vector as it will zero the attention; rather, by a all-one vector
        if not head:
            vec = [1] * (len(heads) + 1)
        res.append(vec)
    return res

def get_dataset_patterns(dataset_dir: Path, min_frequency = path_patterns.MIN_FREQUENCY) -> List[str]:
    """ Retreive the set of pattern labels for a specific dataset (e.g. spcay/laptops_to_device_1).
        This list is used for generating the vocabulary and embedding matrix for auxiliary pattern classificaiton task."""
    csv_df = pd.read_csv(dataset_dir / "train.csv")
    aux_labels = csv_df.AUX_TASK[csv_df.AUX_TASK.notnull()]
    patterns = [aux_lbl.split(AUX_SEP)[1]   # taking only OT->->AT path patterns out of all auxuliary labels   
                for aux_lbl in aux_labels 
                if AUX_SEP in aux_lbl]
    # filter path patterns by frequency
    pattern_counter = Counter(patterns)
    frequent_patterns = [patt for patt, freq in pattern_counter.items() 
                         if freq >= min_frequency]
    return frequent_patterns

def prepare_pattern_info_in_cfg(hparams: Namespace):
    # prepare OT->-AT pattern output vocab 
    all_patterns: List[str] = get_dataset_patterns(hparams.csv_dir / hparams.data_dir)
    all_patterns = ["RARE_PATTERN"] + all_patterns
    hparams.all_patterns = all_patterns

def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        hparams: Namespace
    ) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`."""

    label_map = {label: i for i, label in enumerate(label_list)}
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token_id
    pad_token_segment_id = tokenizer.pad_token_type_id
    cls_token_segment_id = 0
    pad_token_label_id = CrossEntropyLoss().ignore_index

    features = []
    for (ex_index, ex) in enumerate(examples):
        if ex_index % 1_000 == 0:
            log.debug("Writing example %d of %d", ex_index, len(examples))
        tokens, label_ids, heads, sub_toks = [], [], [], []
        aux_task_labels = []
        # generate an indexing map - from original word sequence (as sequences in InputFeatures) 
        # into new, longer sequence corresponding to BERT's BPE subwords.  
        # Mapping is from (index of) a word to (index of) its first subword. 
        orig_index2subtok_index, word_i, subword_i = {0:0}, 0, 0  

        for word, label, head, _, sub_tok, aux_task_label in zip(ex.words, ex.labels, ex.heads, ex.pos_tags, ex.sub_toks, ex.aux_task_labels):
            
            heads.append(head)
            sub_toks.append(sub_tok)
            aux_task_labels.append(aux_task_label)
            word_tokens = tokenizer.tokenize(word)
            
            # bert-base-multilingual-cased sometimes output "nothing ([])
            # when calling tokenize with just a space.
            if len(word_tokens) > 0:
                # align `label_ids` and `aux_task_labels` sequences with sub-word tokenization - add elements for splitted sub_tokens
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word,
                # and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                # Same strategy for auxiliary labels - pad further sub-tokens with 'O'
                aux_task_labels.extend(['O'] * (len(word_tokens) - 1))
            
                # for next element in sequence
                word_i += 1; 
                subword_i += len(word_tokens)  
                orig_index2subtok_index[word_i] = subword_i        


        ######### Add syntactic information #################
        binary_heads_orig = binarize_multi_hot(heads) 
        binary_heads = apply_heads_to_subtokens(binary_heads_orig, sub_toks)

        ####################### DEBUG ############################
        # if binary_heads.shape[1] != len(tokens) + 1:
        #     print('$$'*20)
        #     print()
        #     print(f'len(heads): {len(heads)}')
        #     print(f'binary_heads.shape[1]: {binary_heads.shape[1]}')
        #     print(f'len(tokens): {len(tokens)}')
        #     print(f'len(sub_toks): {len(sub_toks)}')
        #     print([[token] for token in tokens])
        #     print([[word] for word in ex.words])
        #     print(sub_toks)
        #     print()
        ############################################################

        assert binary_heads.shape[0] == binary_heads.shape[1] == len(tokens) + 1
        padded_heads = pad_heads(binary_heads)
        #######################################################################################

        # Account for [CLS] and [SEP]
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        segment_ids = [cls_token_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        
        # Auxiliary Task data 
        """ 
        Auxiliary task is a classification task (especially) for Opinion Term tokens - predicting the 
        corresponding Aspect Term token, and the path pattern toward it via the linguistic graph.  
        """
        # read auxiliary labels for this sentence from InputExample
        aux_AT_tok_idxs, aux_path_patterns = [], []
        task_labels_if_path = [(idx, aux_lbl) 
                               for idx, aux_lbl in enumerate(aux_task_labels) 
                               if AUX_SEP in aux_lbl]
        if task_labels_if_path: # list is emplty iff no OT in sentence has a taken path to AT
            aux_task_tokens, aux_path_pattern_data = zip(*task_labels_if_path)
            for AT_and_patt in aux_path_pattern_data:
                tgt_tok, patt = AT_and_patt.split(AUX_SEP) 
                # "translate" tgt_tok - given as word-level index - into subword indexation 
                tgt_tok = orig_index2subtok_index[int(tgt_tok)]
                aux_AT_tok_idxs.append(tgt_tok)
                aux_path_patterns.append(patt)
            # translate patterns (strings) to ids (ints)
            pattern_to_id = lambda patt: hparams.all_patterns.index(patt) \
                if patt in hparams.all_patterns \
                else hparams.all_patterns.index("RARE_PATTERN") 
            aux_pattern_ids = [pattern_to_id(patt) for patt in aux_path_patterns]
            # filter out instances which their OT or AT indices are beyond max_seq_length
            instances = [
                (op_idx, patt_id, at_idx)
                for (op_idx, patt_id, at_idx) in zip(aux_task_tokens, aux_pattern_ids, aux_AT_tok_idxs)
                if op_idx < max_seq_length and at_idx < max_seq_length
            ] 
            aux_task_tokens, aux_pattern_ids, aux_AT_tok_idxs = zip(*instances) if instances else ((), (), ())
        else:
            aux_task_tokens, aux_pattern_ids, aux_AT_tok_idxs = (), (), ()
        # re-format: use masks and full-sequence length instead of num-of-targets sized lists
        aux_task_mask = idxs_to_mask(aux_task_tokens, seq_len=max_seq_length)   # masks the non-target (=non OT) tokens 
        aux_pattern_ids_seq = torch.full((max_seq_length,), pad_token_label_id, dtype=torch.int16)   # first pad all sequence with -100
        # add 1 to index in sentence to account for [CLS] token which is first element in sequence
        for idx, patt_id in zip(aux_task_tokens, aux_pattern_ids):
            aux_pattern_ids_seq[idx+1] = patt_id    
        aux_AT_tok_idxs_seq = torch.full((max_seq_length,), pad_token_label_id, dtype=torch.int16)   # first pad all sequence with -100
        for idx, AT_idx in zip(aux_task_tokens, aux_AT_tok_idxs):
            aux_AT_tok_idxs_seq[idx+1] = AT_idx+1   # also add 1 to AT_idx itself to account for [CLS] preliminary token   

        if ex_index < 5:
            log.debug("*** Example ***")
            log.debug("guid: %s", ex.guid)
            log.debug("tokens: %s", " ".join([str(x) for x in tokens]))
            log.debug("input_ids: %s", " ".join([str(x) for x in input_ids]))
            log.debug("input_mask: %s", " ".join([str(x) for x in input_mask]))
            log.debug("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            log.debug("label_ids: %s", " ".join([str(x) for x in label_ids]))
            log.debug("(aux) pattern_ids: %s", " ".join([str(x) for x in aux_pattern_ids_seq]))
            log.debug("(aux) AT-tok_idxs: %s", " ".join([str(x) for x in aux_AT_tok_idxs_seq]))

        input_features = InputFeatures(input_ids=input_ids, attention_mask=input_mask,
                                      token_type_ids=segment_ids, label_ids=label_ids, 
                                      dep_heads=padded_heads, 
                                      op2at_opinion_mask=aux_task_mask, 
                                      op2at_patterns=aux_pattern_ids_seq,
                                      op2at_tgt_asp_index=aux_AT_tok_idxs_seq)
        features.append(input_features)
    return features

def get_labels(path: str) -> List[str]:
    with open(path, encoding='utf-8') as labels_f:
        return labels_f.read().splitlines()

def detailed_metrics(y_true, y_pred):
    """Calculate the main classification metrics for every label type.

    Args:
        y_true: 2d array. Ground truth (correct) target values.
        y_pred: 2d array. Estimated targets as returned by a classifier.
        digits: int. Number of digits for formatting output floating point values.

    Returns:
        type_metrics: dict of label types and their metrics.
        macro_avg: dict of weighted macro averages for all metrics across label types.
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    metrics = {}
    ps, rs, f1s, s = [], [], [], []
    for type_name, true_entities in d1.items():
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)
        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        metrics[type_name.lower() + '_precision'] = tensor(p)
        metrics[type_name.lower() + '_recall'] = tensor(r)
        metrics[type_name.lower() + '_f1'] = tensor(f1)

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)
    macro_avg = {'macro_precision': tensor(np.average(ps, weights=s)),
                 'macro_recall': tensor(np.average(rs, weights=s)),
                 'macro_f1': tensor(np.average(f1s, weights=s))}
    return metrics, macro_avg

def copy_config(name, dest_dir):
    """Load an experiment configuration from a yaml file."""
    cfg_path = Path(os.path.dirname(os.path.realpath(__file__))) / 'config' / (name + '.yaml')
    os.makedirs(dest_dir, exist_ok=True)
    dest = dest_dir / "config.yaml"
    shutil.copy(cfg_path, dest)

def read_config(name):
    """Load an experiment configuration from a yaml file."""
    cfg_path = Path(os.path.dirname(os.path.realpath(__file__))) / 'config' / (name + '.yaml')
    cfg = Namespace(**load_hparams_from_yaml(str(cfg_path)))
    cfg.tag = '' if not cfg.tag else cfg.tag + '_'
    return cfg

def prepare_config(cfg):
    """Some preliminary preprations of the config - called right after reading it from file. """
    # add `is_cross_domain(data)` function as global config method
    cfg.is_cross_domain = lambda dataset: "_to_" in dataset  # domain-adaptation setting 
    
    splits = cfg.splits
    if isinstance(splits, int):
        splits = list(range(1, splits + 1))
        # to account for the different splits in cross- vs. in-domain settings,
        # `cfg.splits` should be a function that map a dataset (from cfg.data) into split list
        def data_to_splits(dataset: str) -> List[int]:
            if cfg.is_cross_domain(dataset):
                return splits
            else:
                # in-domain settings (vs. cross-domain) have 4 splits anyway (cross validation)
                return list(range(1,5))
        cfg.splits = data_to_splits
    
    if not hasattr(cfg, 'formalism'):   # backward compatible with older configs
        cfg.formalism = 'spacy'         # default formalism is spacy dependency parser
    cfg.csv_dir = LIBERT_DIR / 'data' / 'csv' / cfg.formalism
    # for relation labels
    with open(cfg.csv_dir / "dep_relations.txt", encoding='utf-8') as deprel_f:
        dep_relations = [l.strip() for l in deprel_f.read().splitlines()]
        cfg.DEP_REL_MAP = {rel: i + 1 for i, rel in enumerate(dep_relations)}
        # determine number of dep-relation labels by dep_relations.txt
        cfg.NUM_REL_LABELS = len(dep_relations) + 1
    
    ds = {'l': 'laptops', 'r': 'restaurants', 'd': 'device'}
    if isinstance(cfg.data, str):
        cfg.data = [f'{ds[d[0]]}_to_{ds[d[1]]}' if len(d) < 3 else d for d in cfg.data.split()]
    

def load_config(name):
    """Load an experiment configuration from a yaml file."""
    cfg = read_config(name)
    prepare_config(cfg)
    return cfg

def tabular(dic: dict, title: str) -> str:
    res = "\n\n{}\n".format(title)
    key_vals = [(k, f"{float(dic[k]):.4}") for k in sorted(dic)]
    line_sep = f"{os.linesep}{'-' * 175}"
    columns_per_row = 8
    for i in range(0, len(key_vals), columns_per_row):
        subset = key_vals[i: i + columns_per_row]
        res += line_sep + f"{os.linesep}"
        res += f"{subset[0][0]:<10s}\t|  " + "\t|  ".join((f"{k:<13}" for k, _ in subset[1:]))
        res += line_sep + f"{os.linesep}"
        res += f"{subset[0][1]:<10s}\t|  " + "\t|  ".join((f"{v:<13}" for _, v in subset[1:]))
        res += line_sep + "\n"
    res += os.linesep
    return res

def set_as_latest(log_dir):
    log_root = LIBERT_DIR / 'logs'
    link = log_root / 'latest'
    try:
        link.unlink()
    except FileNotFoundError:
        pass
    os.symlink(log_dir, link, target_is_directory=True)

def write_summary_tables(cfg, exp_id, log_dir, sig_result):

    filename = f'{exp_id}'
    with open(log_dir / f'{filename}.csv', 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ['']
        sub_header = ['Seeds Average']
        # split cfg.data into cross-domain datasets and in-domain datasets
        cross_datasets = [data for data in cfg.data if cfg.is_cross_domain(data)]
        indomain_datasets = [data for data in cfg.data if not cfg.is_cross_domain(data)]
        
        is_baseline = True in cfg.baseline
        is_model = False in cfg.baseline
        # reporting significant test (model vs. baseline) only for cross domains
        is_sig = bool(cross_datasets) and is_baseline and is_model
        
        # set header and subheader 
        if cross_datasets:
            for dataset in cross_datasets:
                ds_split = dataset.split('_')
                ds_str = f"{ds_split[0][0]}-->{ds_split[2][0]}".upper()
                header.extend([ds_str, f'{ds_str} '])
            header.extend(['Cross-Domain', 'Cross-Domain'])
            header.extend(['Cross-Domain', 'Cross-Domain'] if is_sig else ['', ''])
            sub_header += ['AS', 'OP'] * len(cross_datasets) + ['ASP_MEAN', 'OP_MEAN'] + ['', '']
        if indomain_datasets:
            header += ['|']
            for dataset in indomain_datasets:
                ds_str = f"{dataset[0]}-->{dataset[0]}".upper()
                header.extend([ds_str, f'{ds_str} '])
            header.extend(['In-Domain', 'In-Domain'])
            sub_header += ['|'] + ['AS', 'OP']*len(indomain_datasets) + ['ASP_MEAN', 'OP_MEAN']
       
        csv_writer.writerow(header)
        csv_writer.writerow(sub_header)

        # Seeds Average rows - baseline, model, delta
        def compute_model_agg_rows_on_datasets(datasets: List[str], model_name):
            model_row, model_std_row = [], []
            for dataset in datasets:
                dataset_dir = log_dir / dataset
                model_dir = dataset_dir / f'{model_name}_AGGREGATED_{model_name}_test' / 'csv'

                model_mean_asp, model_std_asp = get_score_from_csv_agg(model_dir, 'asp_f1')
                model_mean_op, model_std_op = get_score_from_csv_agg(model_dir, 'op_f1')

                model_row.extend([model_mean_asp, model_mean_op])
                model_std_row.extend([model_std_asp, model_std_op])
            # Compute Means across datasets, and append to row
            asp_mean = np.mean([model_row[i] for i in range(0, len(model_row), 2)] or -100)
            op_mean = np.mean([model_row[i] for i in range(1, len(model_row), 2)] or -100)
            model_row.extend([asp_mean, op_mean])
            return model_row, model_std_row
        
        # `cd` for cross-domain, `id` for in-domain 
        if is_baseline:
            cd_baseline_row, cd_baseline_std_row = \
                compute_model_agg_rows_on_datasets(cross_datasets, f'{cfg.model_type}_baseline') 
            id_baseline_row, id_baseline_std_row = \
                compute_model_agg_rows_on_datasets(indomain_datasets, f'{cfg.model_type}_baseline') 
        if is_model:
            cd_model_row, cd_model_std_row = \
                compute_model_agg_rows_on_datasets(cross_datasets, f'{cfg.model_type}') 
            id_model_row, id_model_std_row = \
                compute_model_agg_rows_on_datasets(indomain_datasets, f'{cfg.model_type}') 


        format_list = lambda means, stds: [f'{m:.2f} ({s:.2f})' for m, s in zip(means, stds)]
        # prepare full baseline\model\delta rows - including cross-domain and in-domain 
        def write_full_row(name, cd_model_row, cd_model_std_row, id_model_row, id_model_std_row):
            model_row = [name] + (format_list(cd_model_row[:-2], cd_model_std_row) + cd_model_row[-2:] + ['', ''] if cross_datasets else [])
            if indomain_datasets:
                model_row += ['|'] + format_list(id_model_row[:-2], id_model_std_row) + id_model_row[-2:]
            csv_writer.writerow(model_row)
        
        if is_baseline:
            write_full_row("baseline", cd_baseline_row, cd_baseline_std_row, id_baseline_row, id_baseline_std_row)
        if is_model:
            write_full_row("model", cd_model_row, cd_model_std_row, id_model_row, id_model_std_row)
               
        # compute Delta 
        if is_baseline and is_model:
            cd_deltas_row = np.array(cd_model_row) - np.array(cd_baseline_row)
            id_deltas_row = np.array(id_model_row) - np.array(id_baseline_row)
            deltas_row = ['delta'] + ([f'{d:.2f}' for d in cd_deltas_row] + ['', ''] if cross_datasets else [])
            if indomain_datasets:
                deltas_row += ['|'] + [f'{d:.2f}' for d in id_deltas_row]
            csv_writer.writerow(deltas_row)

        # Compute and write per-seed results
        
        def compute_model_seed_rows_on_datasets(seed: int, datasets: List[str], baseline: bool):
            model_name = f'{cfg.model_type}_baseline' if baseline else cfg.model_type            
            model_row, model_std_row = [], []
            for dataset in datasets:
                model_asp, model_op = [], []
                for split in cfg.splits(dataset):
                    dataset_dir = log_dir / dataset
                    model_csv = dataset_dir / f'{model_name}_seed_{seed}_split_{split}_test'\
                        / f'version_{model_name}' / 'metrics.csv'

                    model_asp.append(get_score_from_csv(model_csv, 'asp_f1'))
                    model_op.append(get_score_from_csv(model_csv, 'op_f1'))

                model_row.extend([np.array(model_asp).mean(), np.array(model_op).mean()])
                model_std_row.extend([np.array(model_asp).std(), np.array(model_op).std()])
            # Compute Means
            asp_mean = np.mean([model_row[i] for i in range(0, len(model_row), 2)] or -100)
            op_mean = np.mean([model_row[i] for i in range(1, len(model_row), 2)] or -100)
            model_row.extend([asp_mean, op_mean])
            return model_row, model_std_row

        # prepare significance test results (only for cross-domain)

        if is_sig:
            _, _, all_alphas_scores = sig_result

        for i, seed in enumerate(cfg.seeds):
            if is_sig:
                alpha_001_sig = all_alphas_scores[1][i]
                alpha_005_sig = all_alphas_scores[2][i]
                sig_str = [f'{v:.2f}' for v in [alpha_001_sig, alpha_005_sig]]

            csv_writer.writerow([''] * len(header))

            # Write Seed header 
            seed_header = [f'Seed {seed}'] + (['AS', 'OP'] * len(cross_datasets) + ['ASP_MEAN', 'OP_MEAN'] \
                 + (['SIGNIFICANCE @ p=0.01', 'SIGNIFICANCE @ p=0.05'] if is_sig else ['', '']) \
                    if cross_datasets else [])
            if indomain_datasets:
                seed_header += ['|'] + ['AS', 'OP'] * len(indomain_datasets) + ['ASP_MEAN', 'OP_MEAN']
            csv_writer.writerow(seed_header)
            
            def write_full_row(model_name, cd_model_row, cd_model_std_row, id_model_row, id_model_std_row):
                # prepare full nodel/baseline row - including cross-domain and in-domain 
                model_row = [model_name] + (format_list(cd_model_row[:-2], cd_model_std_row) + cd_model_row[-2:] + (sig_str if is_sig else ['', '']))
                if indomain_datasets:
                    model_row += ['|'] + format_list(id_model_row[:-2], id_model_std_row) + id_model_row[-2:]
                csv_writer.writerow(model_row)

            # Compute per-seed results - `cd` for cross-domain, `id` for in-domain
            
            if is_baseline: 
                cd_baseline_row, cd_baseline_std_row = \
                    compute_model_seed_rows_on_datasets(seed, cross_datasets, baseline=True) 
                id_baseline_row, id_baseline_std_row = \
                    compute_model_seed_rows_on_datasets(seed, indomain_datasets, baseline=True)
                write_full_row("baseline", cd_baseline_row, cd_baseline_std_row, id_baseline_row, id_baseline_std_row)

            if is_model: 
                cd_model_row, cd_model_std_row = \
                    compute_model_seed_rows_on_datasets(seed, cross_datasets, baseline=False) 
                id_model_row, id_model_std_row = \
                    compute_model_seed_rows_on_datasets(seed, indomain_datasets, baseline=False)
                write_full_row("model", cd_model_row, cd_model_std_row, id_model_row, id_model_std_row)
            
            # Compute delta
            if is_baseline and is_model:
                cd_deltas_row = np.array(cd_model_row) - np.array(cd_baseline_row)
                id_deltas_row = np.array(id_model_row) - np.array(id_baseline_row)
                # prepare full delta row - including cross-domain and in-domain 
                deltas_row = ['delta'] + (([f'{d:.2f}' for d in cd_deltas_row] + ['', '']) if cross_datasets else [])
                if indomain_datasets:
                    deltas_row += ['|'] + [f'{d:.2f}' for d in id_deltas_row]
                csv_writer.writerow(deltas_row)


def get_score_from_csv(csv_file, metric):
    with open(csv_file, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        metric_idx = rows[0].index(metric)
        metric_val = float(rows[1][metric_idx]) * 100
        return metric_val

def get_score_from_csv_agg(csv_dir, metric):
    with open(csv_dir / f'{metric}.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        mean = float(rows[1][1]) * 100
        std = float(rows[1][2]) * 100
        return mean, std

def pretty_datetime():
    return dt.now().strftime("%a_%b_%d_%H:%M:%S")


if __name__ == "__main__":
    significance(load_config('model'), LIBERT_DIR / 'logs', 'Thu_Jul_30_00:43:52')
