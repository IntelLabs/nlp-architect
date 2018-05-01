# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
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
# ******************************************************************************

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import


def label_precision_recall_f1(truth, preds, labels):
    """
    Get precision/recall/f1 scores of given truth and predicted labels.
    Calculate metrics only of given list of labels.
    """
    fp = 0.
    fn = 0.
    tp = 0.
    for t, p in zip(truth, preds):
        for t_i, p_i in zip(t, p):
            if t_i in labels and p_i in labels and t_i == p_i:
                tp += 1.
            elif t_i in labels and p_i not in labels or \
                    t_i in labels and p_i in labels and t_i != p_i:
                fn += 1.
            elif t_i not in labels and p_i in labels:
                fp += 1.
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def extract_nps(text, annotation):
    """
    Extract Noun Phrases from given text tokens and phrase annotations
    """
    np_starts = [i for i in range(len(annotation)) if annotation[i] == 'B-NP']
    np_indexes = []
    for s in np_starts:
        i = 1
        while s+i < len(annotation) and annotation[s + i] == 'I-NP':
            i += 1
        np_indexes.append((s, s + i))
    return [' '.join(text[s:e]) for s, e in np_indexes]
