# ******************************************************************************
# Copyright 2014-2018 Intel Corporation
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
# pylint: disable=deprecated-module
from __future__ import unicode_literals, print_function, division, \
    absolute_import

import re
import io
from collections import Counter
from optparse import OptionParser


class ConllEntry:
    def __init__(self, eid, form, lemma, pos, cpos, feats=None, parent_id=None,
                 relation=None, deps=None, misc=None):
        self.id = eid
        self.form = form
        self.norm = normalize(form)
        self.cpos = cpos.upper()
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

        self.vec = None
        self.lstms = None

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos,
                  self.feats,
                  str(self.pred_parent_id) if self.pred_parent_id is not None
                  else None,
                  self.pred_relation, self.deps,
                  self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


def vocab(conll_path):
    words_count = Counter()
    pos_count = Counter()
    rel_count = Counter()

    for sentence in read_conll(conll_path):
        words_count.update(
            [node.norm for node in sentence if isinstance(node, ConllEntry)])
        pos_count.update(
            [node.pos for node in sentence if isinstance(node, ConllEntry)])
        rel_count.update([node.relation for node in sentence if
                          isinstance(node, ConllEntry)])

    return words_count, {w: i for i, w in enumerate(words_count.keys())}, list(
        pos_count.keys()), list(rel_count.keys())


def read_conll(conll_path):
    with io.open(conll_path, 'r') as conll_fp:
        root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_',
                          -1, 'rroot', '_', '_')
        tokens = [root]
        for line in conll_fp:
            stripped_line = line.strip()
            tok = stripped_line.split('\t')
            if not tok or line.strip() == '':
                if len(tokens) > 1:
                    yield tokens
                tokens = [root]
            else:
                if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                    # noinspection PyTypeChecker
                    tokens.append(stripped_line)
                else:
                    tokens.append(
                        ConllEntry(int(tok[0]), tok[1], tok[2], tok[4], tok[3],
                                   tok[5],
                                   int(tok[6]) if tok[6] != '_' else -1,
                                   tok[7], tok[8], tok[9]))
        if len(tokens) > 1:
            yield tokens


def write_conll(filename, conll_gen):
    with io.open(filename, 'w') as file:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                file.write(str(entry) + '\n')
            file.write('\n')


NUMBER_REGEX = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")


def normalize(word):
    return 'NUM' if NUMBER_REGEX.match(word) else word.lower()


def get_option_parser(kwargs):
    opt_parser = OptionParser()
    opt_parser.add_option("--model", dest="model", help="Load model file",
                          metavar="FILE", default=kwargs.get('model'))
    opt_parser.add_option("--wembedding", type="int", dest="wembedding_dims",
                          default=kwargs.get('wembedding', 100))
    opt_parser.add_option("--pembedding", type="int", dest="pembedding_dims",
                          default=kwargs.get('pembedding', 25))
    opt_parser.add_option("--rembedding", type="int", dest="rembedding_dims",
                          default=kwargs.get('rembedding', 25))
    opt_parser.add_option("--epochs", type="int", dest="epochs",
                          default=kwargs.get('epochs', 30))
    opt_parser.add_option("--hidden", type="int", dest="hidden_units",
                          default=kwargs.get('hidden', 100))
    opt_parser.add_option("--hidden2", type="int", dest="hidden2_units",
                          default=kwargs.get('hidden2', 0))
    opt_parser.add_option("--lr", type="float", dest="learning_rate",
                          default=kwargs.get('lr', 0.1))
    opt_parser.add_option("--outdir", type="string", dest="output",
                          default=kwargs.get('outdir'))
    opt_parser.add_option("--activation", type="string", dest="activation",
                          default=kwargs.get('activation', 'tanh'))
    opt_parser.add_option("--lstmlayers", type="int", dest="lstm_layers",
                          default=kwargs.get('lstmlayers', 2))
    opt_parser.add_option("--lstmdims", type="int", dest="lstm_dims",
                          default=125)
    opt_parser.add_option("--disable_blstm", action="store_false",
                          dest="blstmFlag",
                          default=kwargs.get('disable_blstm', True))
    opt_parser.add_option("--disable_labels", action="store_false",
                          dest="labelsFlag",
                          default=kwargs.get('disable_labels', True))
    opt_parser.add_option("--disable_bibi_lstm", action="store_false",
                          dest="bibiFlag",
                          default=kwargs.get('disable_bibi_lstm', True))
    opt_parser.add_option("--disable_costaug", action="store_false",
                          dest="costaugFlag",
                          default=kwargs.get('disable_costaug', True))
    opt_parser.add_option("--dynet_seed", type="int", dest="seed", default=0)
    opt_parser.add_option("--dynet_mem", type="int", dest="mem", default=0)
    return opt_parser
