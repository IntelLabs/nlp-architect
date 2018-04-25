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
from __future__ import unicode_literals, print_function, division, \
    absolute_import

import os
import pickle
import io
import subprocess
from models.bist.bmstparser import parser_utils
from models.bist.bmstparser.mstlstm import MSTParserLSTM
from utils.evaluation_script.conll17_ud_eval import run_conllu_eval


class BISTTrain:
    """
        Args:
        kwargs (:obj:`dict`, optional): Command-line arguments to override
            (used when invoked from python), including:
            `outdir` (str) - Path to output directory.
            `train` (str) - Path to .conll training file.
            `dev` (str) - Path to .conll development file.
            `epochs` (int) - Number of epochs (iterations) to perform.
            (see README for more option parameters)

        Attributes:
            options (Values): Model parameters.
            write_path (str): Path to write predictions output file to.
            parser_model (MSTParserLSTM): The BIST LSTM model to use for inference.
    """

    def __init__(self, kwargs=None):
        if not kwargs:
            kwargs = {}

        print('\nInitializing BISTTrain...\n')
        self.options = self.get_input_params(kwargs)
        self.is_conllu = os.path.splitext(self.options.conll_dev.lower())[1] == '.conllu'
        self.parser_model = self.init_model()

    def init_model(self):
        """Initializes a model with specified parameters."""
        print('Preparing vocab...')
        words, w2i, pos, rels = parser_utils.vocab(self.options.conll_train)
        print('Finished collecting vocab.')
        parser_params = words, w2i, pos, rels, self.options
        with io.open(os.path.join(self.options.output, 'params.pickle'), 'wb') as file:
            pickle.dump(parser_params, file)
        print('Initializing BIST MSTParserLSTM...')
        model = MSTParserLSTM(words, w2i, pos, rels, self.options)
        print('Done.')
        return model

    def run(self):
        """
        Returns:
            model_path (str): Path to the genrated .model file.
        """
        model_path = None
        for epoch in range(self.options.epochs):
            print('Starting epoch', epoch + 1)
            epoch_str = '_epoch_' + str(epoch + 1)
            self.parser_model.train(self.options.conll_train)
            dev_path = os.path.join(self.options.output, 'dev' + epoch_str + '.conll' +
                                    ('u' if self.is_conllu else ''))
            dev_prediction = self.parser_model.predict(self.options.conll_dev)
            parser_utils.write_conll(dev_path, dev_prediction)
            model_path = os.path.abspath(os.path.join(
                self.options.output, 'bist' + epoch_str + '.model'))
            self.parser_model.save(model_path)
            self.run_eval(dev_path)
        return model_path

    @staticmethod
    def get_input_params(kwargs):
        """
        Args:
            kwargs (:obj:`dict`, optional): Command-line arguments to override
                (used when invoked from python), including:
                `outdir` (str) - Path to output directory.
                `train` (str) - Path to .conll training file.
                `dev` (str) - Path to .conll development file.
                `epochs` (int) - Number of epochs (iterations) to perform.
                (see README for more option parameters)

        Returns:
            A 2-tuple consisting of model options and working directory.
        """
        opt_parser = parser_utils.get_option_parser(kwargs)
        opt_parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file",
                              metavar="FILE", default=kwargs.get('train'))
        opt_parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file",
                              metavar="FILE", default=kwargs.get('dev'))
        options = opt_parser.parse_args()[0]

        return options

    def run_eval(self, dev_path):
        """Evaluates the result using the appropriate script."""
        if self.is_conllu:
            run_conllu_eval(gold_file=self.options.conll_dev, test_file=dev_path,
                            verbose=True)
        else:
            eval_script_path = \
                os.path.join(os.path.dirname(os.path.realpath(__file__)), 'utils', 'eval.pl')

            with open(dev_path + '.txt', 'w') as out_file:
                subprocess.run(['perl', eval_script_path, '-g', self.options.conll_dev, '-s',
                                dev_path], stdout=out_file)
