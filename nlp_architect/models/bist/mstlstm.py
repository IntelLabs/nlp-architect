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
# pylint: disable=no-name-in-module
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches

import random
import time
from collections import namedtuple
from operator import itemgetter

from dynet import ParameterCollection, AdamTrainer, LSTMBuilder, tanh, logistic, rectify, cmult, \
    SimpleRNNBuilder, concatenate, np, renew_cg, esum

from nlp_architect.data.conll import ConllEntry
from nlp_architect.models.bist import decoder
from nlp_architect.models.bist.utils import read_conll


# Things that were changed from the original:
# - Added input validation
# - Updated function and object names to dyNet 2.0.2 and Python 3
# - Removed external embeddings option
# - Reformatted code and variable names to conform with PEP8
# - Added dict_to_obj()
# - Added option for train() to get ConllEntry input
# - Added legal header
# - Disabled some style checks


class MSTParserLSTM(object):
    """Underlying LSTM model for MSTParser used by BIST parser."""

    def __init__(self, vocab, w2i, pos, rels, options):
        if isinstance(options, dict):
            options = _dict_to_obj(options, 'Values')

        self.model = ParameterCollection()
        random.seed(1)
        self.trainer = AdamTrainer(self.model)

        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify,
                            'tanh3': (lambda x: tanh(cmult(cmult(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.blstm_flag = options.blstmFlag
        self.labels_flag = options.labelsFlag
        self.costaug_flag = options.costaugFlag
        self.bibi_flag = options.bibiFlag

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.layers = options.lstm_layers
        self.words_count = vocab
        self.vocab = {word: ind + 3 for word, ind in list(w2i.items())}
        self.pos = {word: ind + 3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        if self.bibi_flag:
            self.builders = [LSTMBuilder(1, self.wdims + self.pdims, self.ldims, self.model),
                             LSTMBuilder(1, self.wdims + self.pdims, self.ldims, self.model)]
            self.bbuilders = [LSTMBuilder(1, self.ldims * 2, self.ldims, self.model),
                              LSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]
        elif self.layers > 0:
            self.builders = \
                [LSTMBuilder(self.layers, self.wdims + self.pdims, self.ldims, self.model),
                 LSTMBuilder(self.layers, self.wdims + self.pdims, self.ldims, self.model)]
        else:
            self.builders = [SimpleRNNBuilder(1, self.wdims + self.pdims, self.ldims, self.model),
                             SimpleRNNBuilder(1, self.wdims + self.pdims, self.ldims, self.model)]

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims))
        self.plookup = self.model.add_lookup_parameters((len(pos) + 3, self.pdims))
        self.rlookup = self.model.add_lookup_parameters((len(rels), self.rdims))

        self.hid_layer_foh = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hid_layer_fom = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hid_bias = self.model.add_parameters((self.hidden_units))

        self.hid2_layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.hid2_bias = self.model.add_parameters((self.hidden2_units))

        self.out_layer = self.model.add_parameters(
            (1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))

        if self.labels_flag:
            self.rhid_layer_foh = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
            self.rhid_layer_fom = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
            self.rhid_bias = self.model.add_parameters((self.hidden_units))
            self.rhid2_layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
            self.rhid2_bias = self.model.add_parameters((self.hidden2_units))
            self.rout_layer = self.model.add_parameters(
                (len(self.irels),
                 self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
            self.rout_bias = self.model.add_parameters((len(self.irels)))

    def _get_expr(self, sentence, i, j):
        # pylint: disable=missing-docstring
        if sentence[i].headfov is None:
            sentence[i].headfov = self.hid_layer_foh.expr() * concatenate(
                [sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].modfov is None:
            sentence[j].modfov = self.hid_layer_fom.expr() * concatenate(
                [sentence[j].lstms[0], sentence[j].lstms[1]])

        if self.hidden2_units > 0:
            output = \
                self.out_layer.expr() * self.activation(
                    self.hid2_bias.expr() + self.hid2_layer.expr() * self.activation(
                        sentence[i].headfov + sentence[j].modfov
                        + self.hid_bias.expr()))  # + self.outBias
        else:
            output = self.out_layer.expr() * self.activation(
                sentence[i].headfov + sentence[j].modfov + self.hid_bias.expr())  # + self.outBias
        return output

    def _evaluate(self, sentence):
        # pylint: disable=missing-docstring
        exprs = [[self._get_expr(sentence, i, j) for j in range(len(sentence))]
                 for i in range(len(sentence))]
        scores = np.array([[output.scalar_value() for output in exprsRow] for exprsRow in exprs])
        return scores, exprs

    def _evaluate_label(self, sentence, i, j):
        # pylint: disable=missing-docstring
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = self.rhid_layer_foh.expr() * concatenate(
                [sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].rmodfov is None:
            sentence[j].rmodfov = self.rhid_layer_fom.expr() * concatenate(
                [sentence[j].lstms[0], sentence[j].lstms[1]])

        if self.hidden2_units > 0:
            output = self.rout_layer.expr() * self.activation(
                self.rhid2_bias.expr() + self.rhid2_layer.expr()
                * self.activation(sentence[i].rheadfov + sentence[j].rmodfov
                                  + self.rhid_bias.expr())) + self.rout_bias.expr()
        else:
            output = self.rout_layer.expr() * self.activation(
                sentence[i].rheadfov + sentence[j].rmodfov
                + self.rhid_bias.expr()) + self.rout_bias.expr()
        return output.value(), output

    def predict(self, conll_path=None, conll=None):
        # pylint: disable=missing-docstring
        if conll is None:
            conll = read_conll(conll_path)

        for sentence in conll:
            conll_sentence = [entry for entry in sentence if
                              isinstance(entry, ConllEntry)]

            for entry in conll_sentence:
                wordvec = self.wlookup[int(
                    self.vocab.get(entry.norm, 0))] if self.wdims > 0 else None
                posvec = self.plookup[
                    int(self.pos[entry.pos])] if self.pdims > 0 else None
                entry.vec = concatenate(
                    [_f for _f in [wordvec, posvec, None] if _f])

                entry.lstms = [entry.vec, entry.vec]
                entry.headfov = None
                entry.modfov = None

                entry.rheadfov = None
                entry.rmodfov = None

            if self.blstm_flag:
                lstm_forward = self.builders[0].initial_state()
                lstm_backward = self.builders[1].initial_state()

                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    lstm_forward = lstm_forward.add_input(entry.vec)
                    lstm_backward = lstm_backward.add_input(rentry.vec)

                    entry.lstms[1] = lstm_forward.output()
                    rentry.lstms[0] = lstm_backward.output()

                if self.bibi_flag:
                    for entry in conll_sentence:
                        entry.vec = concatenate(entry.lstms)

                    blstm_forward = self.bbuilders[0].initial_state()
                    blstm_backward = self.bbuilders[1].initial_state()

                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        blstm_forward = blstm_forward.add_input(entry.vec)
                        blstm_backward = blstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = blstm_forward.output()
                        rentry.lstms[0] = blstm_backward.output()

            scores, _ = self._evaluate(conll_sentence)
            heads = decoder.parse_proj(scores)

            for entry, head in zip(conll_sentence, heads):
                entry.pred_parent_id = head
                entry.pred_relation = '_'

            dump = False

            if self.labels_flag:
                for modifier, head in enumerate(heads[1:]):
                    scores, _ = self._evaluate_label(conll_sentence, head, modifier + 1)
                    conll_sentence[modifier + 1].pred_relation = \
                        self.irels[max(enumerate(scores), key=itemgetter(1))[0]]

            renew_cg()
            if not dump:
                yield sentence

    def train(self, conll_path):
        # pylint: disable=invalid-name
        # pylint: disable=missing-docstring
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()

        shuffled_data = list(read_conll(conll_path))
        random.shuffle(shuffled_data)
        errs = []
        lerrs = []
        i_sentence = 0

        for sentence in shuffled_data:
            if i_sentence % 100 == 0 and i_sentence != 0:
                print('Processing sentence number:', i_sentence, 'Loss:',
                      eloss / etotal, 'Errors:',
                      (float(eerrors)) / etotal, 'Time', time.time() - start)
                start = time.time()
                eerrors = 0
                eloss = 0.0
                etotal = 0

            conll_sentence = [entry for entry in sentence if isinstance(entry, ConllEntry)]

            for entry in conll_sentence:
                c = float(self.words_count.get(entry.norm, 0))
                drop_flag = (random.random() < (c / (0.25 + c)))
                wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0)) if drop_flag else 0] \
                    if self.wdims > 0 else None
                posvec = self.plookup[int(self.pos[entry.pos])] if self.pdims > 0 else None

                entry.vec = concatenate([_f for _f in [wordvec, posvec, None] if _f])

                entry.lstms = [entry.vec, entry.vec]
                entry.headfov = None
                entry.modfov = None

                entry.rheadfov = None
                entry.rmodfov = None

            if self.blstm_flag:
                lstm_forward = self.builders[0].initial_state()
                lstm_backward = self.builders[1].initial_state()

                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    lstm_forward = lstm_forward.add_input(entry.vec)
                    lstm_backward = lstm_backward.add_input(rentry.vec)

                    entry.lstms[1] = lstm_forward.output()
                    rentry.lstms[0] = lstm_backward.output()

                if self.bibi_flag:
                    for entry in conll_sentence:
                        entry.vec = concatenate(entry.lstms)

                    blstm_forward = self.bbuilders[0].initial_state()
                    blstm_backward = self.bbuilders[1].initial_state()

                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        blstm_forward = blstm_forward.add_input(entry.vec)
                        blstm_backward = blstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = blstm_forward.output()
                        rentry.lstms[0] = blstm_backward.output()

            scores, exprs = self._evaluate(conll_sentence)
            gold = [entry.parent_id for entry in conll_sentence]
            heads = decoder.parse_proj(scores, gold if self.costaug_flag else None)

            if self.labels_flag:
                for modifier, head in enumerate(gold[1:]):
                    rscores, rexprs = self._evaluate_label(conll_sentence, head, modifier + 1)
                    gold_label_ind = self.rels[conll_sentence[modifier + 1].relation]
                    wrong_label_ind = max(((label, scr) for label, scr in enumerate(rscores)
                                           if label != gold_label_ind), key=itemgetter(1))[0]
                    if rscores[gold_label_ind] < rscores[wrong_label_ind] + 1:
                        lerrs.append(rexprs[wrong_label_ind] - rexprs[gold_label_ind])

            e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
            eerrors += e
            if e > 0:
                loss = [(exprs[h][i] - exprs[g][i]) for i, (h, g) in
                        enumerate(zip(heads, gold)) if h != g]  # * (1.0/float(e))
                eloss += e
                mloss += e
                errs.extend(loss)

            etotal += len(conll_sentence)

            if i_sentence % 1 == 0 or errs > 0 or lerrs:
                if errs or lerrs:
                    eerrs = (esum(errs + lerrs))  # * (1.0/(float(len(errs))))
                    eerrs.scalar_value()
                    eerrs.backward()
                    self.trainer.update()
                    errs = []
                    lerrs = []

                renew_cg()

            i_sentence += 1

        if errs:
            eerrs = (esum(errs + lerrs))  # * (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            renew_cg()

        self.trainer.update()
        print("Loss: ", mloss / i_sentence)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.populate(filename)


def _dict_to_obj(dic, name='Object'):
    """Return an object form of a dictionary."""
    return namedtuple(name, dic.keys())(*dic.values())
