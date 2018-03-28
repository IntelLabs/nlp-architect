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
# pylint: disable=deprecated-module
# pylint: disable=invalid-name

from __future__ import unicode_literals, print_function, division, \
    absolute_import

import io
import os.path
import pickle
import sys
import time
from optparse import OptionParser

from models.bist.bmstparser import parser_utils
from models.bist.bmstparser.mstlstm import MSTParserLSTM

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train",
                      help="Annotated CONLL train file", metavar="FILE",
                      default="../data/en-universal-train.conll.ptb")
    parser.add_option("--dev", dest="conll_dev",
                      help="Annotated CONLL dev file", metavar="FILE",
                      default="../data/en-universal-dev.conll.ptb")
    parser.add_option("--test", dest="conll_test",
                      help="Annotated CONLL test file", metavar="FILE",
                      default="../data/en-universal-test.conll.ptb")
    parser.add_option("--extrn", dest="external_embedding",
                      help="External embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file",
                      metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file",
                      metavar="FILE",
                      default="neuralfirstorder.model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims",
                      default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims",
                      default=25)
    parser.add_option("--rembedding", type="int", dest="rembedding_dims",
                      default=25)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string", dest="output",
                      default="results")
    parser.add_option("--activation", type="string", dest="activation",
                      default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers",
                      default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=125)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag",
                      default=True)
    parser.add_option("--disablelabels", action="store_false",
                      dest="labelsFlag", default=True)
    parser.add_option("--predict", action="store_true", dest="predictFlag",
                      default=False)
    parser.add_option("--bibi-lstm", action="store_true", dest="bibiFlag",
                      default=False)
    parser.add_option("--disablecostaug", action="store_false",
                      dest="costaugFlag", default=True)
    parser.add_option("--dynet-seed", type="int", dest="seed", default=0)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)

    (options, args) = parser.parse_args()

    print('Using external embedding:', options.external_embedding)
    root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    if options.predictFlag:
        words, w2i, pos, rels, stored_opt = pickle.load(
            io.open(options.params, 'rb'))

        stored_opt.external_embedding = options.external_embedding

        print('Initializing lstm mstparser:')
        parser = MSTParserLSTM(words, w2i, pos, rels, stored_opt)
        print('Done')

        parser.load(options.model)
        conllu = (os.path.splitext(options.conll_test.lower())[1] == '.conllu')
        tespath = os.path.join(
            options.output,
            'test_pred.conll' if not conllu else 'test_pred.conllu')

        ts = time.time()
        test_res = list(parser.predict(options.conll_test))
        te = time.time()
        print('Finished predicting test.', te - ts, 'seconds.')
        parser_utils.write_conll(tespath, test_res)

        if not conllu:
            os.system(
                'perl %s/utils/eval.pl -g ' % root_dir + options.conll_test +
                ' -s ' + tespath + ' > ' + tespath + '.txt')
        else:
            os.system(
                'python %s/utils/evaluation_script/conll17_ud_eval.py'
                ' -v -w %s/utils/evaluation_script/weights.clas ' %
                (root_dir, root_dir) + options.conll_test + ' ' + tespath +
                ' > ' + tespath + '.txt')
    else:
        print('Preparing vocab')
        words, w2i, pos, rels = parser_utils.vocab(options.conll_train)

        pickle.dump((words, w2i, pos, rels, options),
                    io.open(os.path.join(options.output, options.params),
                            'wb'))
        print('Finished collecting vocab')

        print('Initializing lstm mstparser:')
        parser = MSTParserLSTM(words, w2i, pos, rels, options)

        for epoch in range(options.epochs):
            print('Starting epoch', epoch)
            parser.train(options.conll_train)
            conllu = (
                os.path.splitext(options.conll_dev.lower())[1] == '.conllu')
            devpath = \
                os.path.join(options.output, 'dev_epoch_' + str(epoch + 1) +
                             ('.conll' if not conllu else '.conllu'))
            parser_utils.write_conll(devpath,
                                     parser.predict(options.conll_dev))
            parser.save(os.path.join(options.output,
                                     os.path.basename(options.model) + str(
                                         epoch + 1)))

            if not conllu:
                os.system(
                    'perl %s/utils/eval.pl -g ' % root_dir +
                    options.conll_dev + ' -s '
                    + devpath + ' > ' + devpath + '.txt')
            else:
                os.system(
                    'python %s/utils/evaluation_script/conll17_ud_eval.py'
                    ' -v -w %s/utils/evaluation_script/weights.clas ' %
                    (root_dir, root_dir) + options.conll_dev + ' ' +
                    devpath + ' > ' + devpath + '.txt')
