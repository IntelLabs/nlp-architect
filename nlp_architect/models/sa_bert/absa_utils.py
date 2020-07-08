# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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


import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


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
    # valid_ids: Optional[List[int]] = None

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
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
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
        tokenizer,
        # cls_token_at_end=False,
        # pad_on_left=False,
        # cls_token="[CLS]",
        # sep_token="[SEP]",
        # pad_token=0,
        # sequence_segment_id=0,
        # sep_token_extra=0,
        # cls_token_segment_id=0,
        # pad_token_segment_id=0,
        # mask_padding_with_zero=True,
    ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token
        (0 for BERT, 2 for XLNet)
    """
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token_id
    pad_token_segment_id = tokenizer.pad_token_type_id
    sequence_segment_id = 0
    cls_token_segment_id = 0

    label_map = {label: i for i, label in enumerate(label_list)}
    label_pad = -100 # HF example: label_pad = -100

    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Processing example %d of %d", ex_index, len(examples))
        tokens = []
        labels = []
        # valid_tokens = []
        for token, label in zip(example.words, example.labels):
            new_tokens = tokenizer.tokenize(token)
            tokens.extend(new_tokens)
            # v_tok = [0] * (len(new_tokens))
            # v_tok[0] = 1
            # valid_tokens.extend(v_tok)
            v_lbl = [label_pad] * (len(new_tokens))
            v_lbl[0] = label_map.get(label)
            labels.extend(v_lbl)

        # truncate by max_seq_length
        special_tokens_count = 2
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        # valid_tokens = valid_tokens[: (max_seq_length - special_tokens_count)]
        labels = labels[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        labels += [label_pad]
        # valid_tokens += [0]
        segment_ids = [sequence_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids
        labels = [label_pad] + labels
        # valid_tokens = [0] + valid_tokens
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        labels = labels + ([label_pad] * padding_length)
        # valid_tokens = valid_tokens + ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # assert len(valid_tokens) == max_seq_length
        assert len(labels) == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                label_ids=labels
                #, valid_ids=valid_tokens,
            )
        )
    return features

def get_labels(path: str) -> List[str]:
    with open(path) as f:
        return f.read().splitlines()
