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
import os
import random
import string
import subprocess
import tempfile

from nlp_architect.utils.io import download_file


def run_conlleval(filename):
    """
    Run CoNLL benchmarking script to get global and per
    label accuracy, precision, recall and F1.
    Input is a filename in IOB format:
    <TOKEN> - <TEST LABEL> <PREDICTED LABEL>
    Argument:
        filename(str): tagged file path
    Returns:
        (int, int, int), tuple: Overall precision/recall/f1 and per
        label P/R/F1 in tuple (dictionary format)
    """
    if not os.path.exists('conlleval.pl'):
        download_file('https://www.clips.uantwerpen.be/conll2000/chunking/',
                      'conlleval.txt', 'conlleval.pl')
    proc = subprocess.Popen(['perl', 'conlleval.pl'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename, 'rb').read())
    lines = stdout.decode('utf-8').split('\n')[:-1]
    _, _, _, gp, _, gr, _, gf1 = lines[1].split()
    gp = float(gp[:-2])
    gr = float(gr[:-2])
    gf1 = float(gf1)
    labels = {}
    for l in lines[2:]:
        label, _, p, _, r, _, f1, _ = l.split()
        label = label[:-1]
        p = float(p[:-2])
        r = float(r[:-2])
        f1 = float(f1)
        labels[label] = (p, r, f1)
    return (gp, gr, gf1), labels


def get_conll_scores(predictions, y, y_lex):
    if type(predictions) == list:
        predictions = predictions[-1]
    test_p = predictions.argmax(2)
    test_y = y.argmax(2)

    prediction_data = []
    for n in range(test_y.shape[0]):
        test_yval = [y_lex[i] for i in test_y[n] if i > 0]
        prediction_y = ['O'] * len(test_yval)
        for i, j in enumerate(test_p[n][-len(test_yval):]):
            if j > 0:
                prediction_y[i] = y_lex[j]
        prediction_data.append((test_yval, test_yval, prediction_y))

    temp_fname = tempfile.gettempdir() + \
        os.sep + \
        ''.join([random.choice(string.ascii_letters) for _ in range(10)]) + '__conll_eval'
    with open(temp_fname, 'w') as fp:
        for sample in prediction_data:
            for t, l, p in zip(*sample):
                fp.write('{} - {} {}\n'.format(t, l, p))
            fp.write('\n')

    # run CoNLL benchmark
    scores = run_conlleval(temp_fname)
    os.remove(temp_fname)
    return scores
