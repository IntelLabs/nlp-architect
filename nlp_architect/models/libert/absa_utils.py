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
from enum import Enum
from collections import defaultdict
from typing import List, Optional, Union
from os.path import realpath
from argparse import Namespace
from pathlib import Path
from datetime import datetime as dt

from torch.nn import CrossEntropyLoss
from torch import tensor
from pytorch_lightning import _logger as log
from pytorch_lightning.core.saving import load_hparams_from_yaml
from transformers import PreTrainedTokenizer
from seqeval.metrics.sequence_labeling import get_entities
import numpy as np

from significance import significance_from_cfg as significance

LIBERT_DIR = Path(realpath(__file__)).parent
LOG_ROOT = LIBERT_DIR / 'logs'

DEP_REL_MAP = {rel.strip(): i + 1 for i, rel in enumerate(open(LIBERT_DIR / 'dep_relations.txt', encoding='utf-8'))}

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
    heads: Optional[List[int]]
    head_words: Optional[List[str]]
    syn_rels: Optional[List[str]]
    pos_tags: Optional[List[str]]
    sub_toks: Optional[List[List[str]]]

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
    head_idx: Optional[List[int]] = None
    syn_rels: Optional[List[int]] = None
    
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
        words, labels, heads, head_words, syn_rels, pos_tags, sub_toks = [], [], [], [], [], [], []
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row == empty_row:
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels,
                                                    heads=heads, head_words=head_words, syn_rels=syn_rels, pos_tags=pos_tags, sub_toks=sub_toks))
                    guid_index += 1
                    words, labels, heads, head_words, pos_tags, sub_toks = [], [], [], [], [], []
            else:
                word, label, head, head_word, syn_rel, pos_tag, sub_tok = row
                words.append(word)
                labels.append(label)
                heads.append(int(head))
                head_words.append(head_word)
                syn_rels.append(syn_rel)
                pos_tags.append(pos_tag)
                sub_toks.append(sub_tok.split() if sub_tok else [word])
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels,
                                            heads=heads, head_words=head_words, syn_rels=syn_rels,
                                            pos_tags=pos_tags, sub_toks=sub_toks))
    return examples


def pad_heads(a):
    target = np.zeros((64, 64), float)
    target[:a.shape[0], :a.shape[1]] = a[:64, :64]
    return target

def pad_syn_rels(a):
    target = np.zeros((64), int)
    target[:a.shape[0]] = a[:64]
    return target

def indexify(a):
    res = []
    for row in a:
        idx = np.where(row == 1)[0]
        if idx:
            res.append(idx[0])
        else:
            res.append(0)
    return res

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

def apply_syn_rels_to_subtokens(syn_rels, sub_tokens_list):
    res = [0] ## add null relation for CLS token
    for syn_rel, sub_tokens in zip(syn_rels, sub_tokens_list):
        res.extend([syn_rel] * len(sub_tokens))
    return np.array(res)

def binarize(preds):
    res = []
    for pred in preds:
        res.append([1 if i == pred else 0 for i in range(len(preds) + 1)])
    return res

def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
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
        tokens, label_ids, heads, sub_toks, syn_rels = [], [], [], [], []

        for word, label, head, syn_rel, _, sub_tok in zip(ex.words, ex.labels, ex.heads, ex.syn_rels, ex.pos_tags, ex.sub_toks):
            syn_rels.append(DEP_REL_MAP[syn_rel.lower()])
            heads.append(head)
            sub_toks.append(sub_tok)
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([])
            # when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word,
                # and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        ######### Add syntactic information #################
        binary_heads = apply_heads_to_subtokens(binarize(heads), sub_toks)
        syn_rels = apply_syn_rels_to_subtokens(syn_rels, sub_toks)

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
        padded_syn_rels = pad_syn_rels(syn_rels)

        #######################################################################################

        padded_heads_idx = indexify(padded_heads)
        padded_heads_idx = [(v - i) + 11 if v != 0 and -11 < v - i < 11 else 0 for i, v in enumerate(padded_heads_idx)]
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

        if ex_index < 5:
            log.debug("*** Example ***")
            log.debug("guid: %s", ex.guid)
            log.debug("tokens: %s", " ".join([str(x) for x in tokens]))
            log.debug("input_ids: %s", " ".join([str(x) for x in input_ids]))
            log.debug("input_mask: %s", " ".join([str(x) for x in input_mask]))
            log.debug("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            log.debug("label_ids: %s", " ".join([str(x) for x in label_ids]))

        input_features = InputFeatures(input_ids=input_ids, attention_mask=input_mask,
                                      token_type_ids=segment_ids, label_ids=label_ids, 
                                      dep_heads=padded_heads, head_idx=padded_heads_idx, syn_rels=padded_syn_rels)
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

def load_config(name):
    """Load an experiment configuration from a yaml file."""
    cfg_path = Path(os.path.dirname(os.path.realpath(__file__))) / 'config' / (name + '.yaml')
    cfg = Namespace(**load_hparams_from_yaml(str(cfg_path)))

    if isinstance(cfg.splits, int):
        cfg.splits = list(range(1, cfg.splits + 1))

    cfg.tag = '' if cfg.tag is None else '_' + cfg.tag
    ds = {'l': 'laptops', 'r': 'restaurants', 'd': 'device'}
    cfg.data = [f'{ds[d[0]]}_to_{ds[d[1]]}' if len(d) < 3 else d for d in cfg.data.split()]
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

def write_summary_tables(cfg, exp_id):
    filename = f'summary_table_{exp_id}'
    with open(LOG_ROOT / exp_id / f'{filename}.csv', 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ['']
        for dataset in cfg.data:
            ds_split = dataset.split('_')
            ds_str = f"{ds_split[0][0]}-->{ds_split[2][0]}".upper()
            header.extend([ds_str, f'{ds_str} '])
        sub_header = ['Seeds Average'] + ['AS', 'OP'] * len(cfg.data) + ['ASP_MEAN', 'OP_MEAN']
        csv_writer.writerow(header)
        csv_writer.writerow(sub_header)

        baseline_row, baseline_std_row, model_row, model_std_row = [], [], [], []

        for dataset in cfg.data:
            dataset_dir = LOG_ROOT / exp_id / dataset
            baseline_dir = dataset_dir / 'libert_baseline_AGGREGATED_baseline_test' / 'csv'
            model_dir = dataset_dir / f'libert_AGGREGATED_{exp_id}_test' / 'csv'

            baseline_mean_asp, baseline_std_asp = get_score_from_csv_agg(baseline_dir, 'asp_f1')
            baseline_mean_op, baseline_std_op = get_score_from_csv_agg(baseline_dir, 'op_f1')
            model_mean_asp, model_std_asp = get_score_from_csv_agg(model_dir, 'asp_f1')
            model_mean_op, model_std_op = get_score_from_csv_agg(model_dir, 'op_f1')

            baseline_row.extend([baseline_mean_asp, baseline_mean_op])
            model_row.extend([model_mean_asp, model_mean_op])
            baseline_std_row.extend([baseline_std_asp, baseline_std_op])
            model_std_row.extend([model_std_asp, model_std_op])

        for row in baseline_row, model_row:
            asp_mean = np.mean([row[i] for i in range(0, len(row), 2)])
            op_mean = np.mean([row[i] for i in range(1, len(row), 2)])
            row.extend([asp_mean, op_mean])

        deltas_row = np.array(model_row) - np.array(baseline_row)
        format_list = lambda means, stds: [f'{m:.2f} ({s:.2f})' for m, s in zip(means, stds)]
        csv_writer.writerow(['baseline'] + format_list(baseline_row[:-2], baseline_std_row) + baseline_row[-2:])
        csv_writer.writerow(['model'] + format_list(model_row[:-2], model_std_row) + model_row[-2:])
        csv_writer.writerow(['delta'] + [f'{d:.2f}' for d in deltas_row])

        for seed in cfg.seeds:
            baseline_row, baseline_std_row, model_row, model_std_row = [], [], [], []
            csv_writer.writerow([''] * ((2 * len(cfg.data)) + 3))
            seed_header = [f'Seed {seed}'] + ['AS', 'OP'] * len(cfg.data) + ['ASP_MEAN', 'OP_MEAN']
            csv_writer.writerow(seed_header)

            for dataset in cfg.data:
                base_asp, model_asp, base_op, model_op = [], [], [], []
                for split in cfg.splits:
                    dataset_dir = LOG_ROOT / exp_id / dataset
                    baseline_csv = dataset_dir / f'libert_baseline_seed_{seed}_split_{split}_test'\
                        / 'version_baseline' / 'metrics.csv'
                    model_csv = dataset_dir / f'libert_seed_{seed}_split_{split}_test'\
                        / f'version_{exp_id}' / 'metrics.csv'

                    base_asp.append(get_score_from_csv(baseline_csv, 'asp_f1'))
                    base_op.append(get_score_from_csv(baseline_csv, 'op_f1'))
                    model_asp.append(get_score_from_csv(model_csv, 'asp_f1'))
                    model_op.append(get_score_from_csv(model_csv, 'op_f1'))

                baseline_row.extend([np.array(base_asp).mean(), np.array(base_op).mean()])
                model_row.extend([np.array(model_asp).mean(), np.array(model_op).mean()])
                baseline_std_row.extend([np.array(base_asp).std(), np.array(base_op).std()])
                model_std_row.extend([np.array(model_asp).std(), np.array(model_op).std()])

            for row in baseline_row, model_row:
                asp_mean = np.mean([row[i] for i in range(0, len(row), 2)])
                op_mean = np.mean([row[i] for i in range(1, len(row), 2)])
                row.extend([asp_mean, op_mean])

            deltas_row = np.array(model_row) - np.array(baseline_row)
            format_list = lambda means, stds: [f'{m:.2f} ({s:.2f})' for m, s in zip(means, stds)]
            csv_writer.writerow(['baseline'] + format_list(baseline_row[:-2], baseline_std_row) + baseline_row[-2:])
            csv_writer.writerow(['model'] + format_list(model_row[:-2], model_std_row) + model_row[-2:])
            csv_writer.writerow(['delta'] + [f'{d:.2f}' for d in deltas_row])

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
