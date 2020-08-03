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

# pylint: disable=logging-fstring-interpolation
import os
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
from typing import List, Optional, Union
from os.path import realpath
from argparse import Namespace
from pathlib import Path

from pytorch_lightning import _logger as log
from pytorch_lightning.core.saving import load_hparams_from_yaml
from transformers import PreTrainedTokenizer
from seqeval.metrics.sequence_labeling import get_entities
import numpy as np

from significance import significance_from_cfg as significance

LIBERT_DIR = Path(realpath(__file__)).parent


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
    parse_heads: Optional[List[float]] = None

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line in ('', '\n'):
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}",
                                                 words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split()
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
    return examples

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
    pad_token_label_id = -100

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1_000 == 0:
            log.debug("Writing example %d of %d", ex_index, len(examples))
        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([])
            # when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word,
                # and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

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
            log.debug("guid: %s", example.guid)
            log.debug("tokens: %s", " ".join([str(x) for x in tokens]))
            log.debug("input_ids: %s", " ".join([str(x) for x in input_ids]))
            log.debug("input_mask: %s", " ".join([str(x) for x in input_mask]))
            log.debug("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            log.debug("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids=input_ids, attention_mask=input_mask,
                                      token_type_ids=segment_ids, label_ids=label_ids))
    return features

def get_labels(path: str) -> List[str]:
    with open(path) as labels_f:
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

        metrics[type_name.lower() + '_precision'] = p
        metrics[type_name.lower() + '_recall'] = r
        metrics[type_name.lower() + '_f1'] = f1

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)
    macro_avg = {'macro_precision': np.average(ps, weights=s),
                 'macro_recall': np.average(rs, weights=s), 'macro_f1': np.average(f1s, weights=s)}
    return metrics, macro_avg


def load_config(name):
    """Load an experiment configuration from a yaml file."""
    configs_dir = Path(os.path.dirname(os.path.realpath(__file__))) / 'config'
    config = Namespace(**load_hparams_from_yaml(configs_dir / (name + '.yaml')))

    config.tag = '' if config.tag is None else '_' + config.tag
    ds = {'l': 'laptops', 'r': 'restaurants', 'd': 'device'}
    config.data = [f'{ds[d[0]]}_to_{ds[d[1]]}' if len(d) < 3 else d for d in config.data.split()]
    return config

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


if __name__ == "__main__":
    significance(load_config('model'), LIBERT_DIR / 'logs', 'Thu_Jul_30_00:43:52')
