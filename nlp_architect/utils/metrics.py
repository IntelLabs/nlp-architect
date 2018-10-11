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

from __future__ import division, print_function, unicode_literals, absolute_import

from seqeval.metrics import classification_report


def get_conll_scores(predictions, y, y_lex, unk='O'):
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
