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

from nlp_architect.utils.conlleval import evaluate, metrics


def run_conlleval(data):
    """
    Run conlleval python script on given data stream

    Returns:
        A tuple of global P/R/F1
        A dict of P/R/F1 per label
    """
    counts = evaluate(data)
    overall, by_type = metrics(counts)
    overall_scores = (100. * overall.prec, 100. * overall.rec, 100. * overall.fscore)

    by_type_res = {}
    for i, m in sorted(by_type.items()):
        by_type_res[i] = (100. * m.prec, 100. * m.rec, 100. * m.fscore)
    return overall_scores, by_type_res


def get_conll_scores(predictions, y, y_lex):
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
        test_yval = [y_lex[i] for i in test_y[n] if i > 0]
        prediction_y = ['O'] * len(test_yval)
        for i, j in enumerate(test_p[n][-len(test_yval):]):
            if j > 0:
                prediction_y[i] = y_lex[j]
        prediction_data.append((test_yval, test_yval, prediction_y))

    data = []
    for s in prediction_data:
        for t, l, p in zip(*s):
            data.append('{} {} {}\n'.format(t, l, p))
        data.append('\n')
    return run_conlleval(data)
