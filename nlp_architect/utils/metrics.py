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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score


def get_conll_scores(predictions, y, y_lex, unk='O'):
    """Get Conll style scores (precision, recall, f1)
    """
    if isinstance(predictions, list):
        predictions = predictions[-1]
    test_p = predictions
    if len(test_p.shape) > 2:
        test_p = test_p.argmax(2)
    test_y = y
    if len(test_y.shape) > 2:
        test_y = test_y.argmax(2)

    prediction_data = []
    for n in range(test_y.shape[0]):
        test_yval = []
        for i in list(test_y[n]):
            try:
                test_yval.append(y_lex[i])
            except KeyError:
                pass
        test_pval = [unk] * len(test_yval)
        for e, i in enumerate(list(test_p[n])[:len(test_pval)]):
            try:
                test_pval[e] = y_lex[i]
            except KeyError:
                pass
        prediction_data.append((test_yval, test_pval))
    y_true, y_pred = list(zip(*prediction_data))
    return classification_report(y_true, y_pred, digits=3)


def simple_accuracy(preds, labels):
    """return simple accuracy
    """
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    """return accuracy and f1 score
    """
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    """get pearson and spearman correlation
    """
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def tagging(preds, labels):
    p = precision_score(labels, preds)
    r = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return p, r, f1