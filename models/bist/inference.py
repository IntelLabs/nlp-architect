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
import pathlib
import pickle
import sys
import time
import io
from bmstparser import parser_utils
from bmstparser.mstlstm import MSTParserLSTM


class BISTInference:
    """
        Args:
            kwargs (:obj:`dict`, optional): Command-line arguments to override
                (used when invoked from python):
                `input` (str) - Path to .conll input file.
                `outdir` (str) - Path to output directory.
                `model` (str) - Path to .model file to load the model from.
                `eval` (bool) - Specifies whether to generate an evaluation file.

        Attributes:
            options (Values): Parameters for model.
            root_dir (str): Path to execution root directory
            write_path (str): Path to write predictions output file to
            parser_model (MSTParserLSTM): The BIST LSTM model to use for inference
    """

    def __init__(self, kwargs=None):
        if not kwargs:
            kwargs = {}

        print('\nInitializing BISTInference...')
        self.options, self.root_dir, self.write_path = self.get_input_params(kwargs)
        self.parser_model = self.load_model(self.options.model)

    def run(self):
        """
        Returns:
            inference_res (list of ConllEntry): The list of input
                sentences with predicted dependencies attached. A sentence
                is represented by a list of annotated tokens (ConllEntry).
        """
        print('\nExecuting BIST inference on ' + str(self.options.conll_input) + '...\n')
        start_time = time.time()
        inference_res = list(self.parser_model.predict(self.options.conll_input))
        end_time = time.time()
        print('Finished inference in ', end_time - start_time, 'seconds.')
        parser_utils.write_conll(self.write_path, inference_res)

        if self.options.evalFlag:
            self.run_eval()
        return inference_res

    def run_eval(self):
        """Evaluates the result using the appropriate script."""
        if self.write_path.endswith('.conllu'):
            os.system('python %s/utils/evaluation_script/conll17_ud_eval.py'
                      ' -v -w %s/utils/evaluation_script/weights.clas '
                      % (self.root_dir, self.root_dir) + self.options.conll_input +
                      ' ' + self.write_path + ' > ' + self.write_path + '.txt')
        else:
            os.system('perl %s/utils/eval.pl -g ' % self.root_dir +
                      self.options.conll_input + ' -s ' + self.write_path + ' > ' +
                      self.write_path + '.txt')

    @staticmethod
    def get_input_params(kwargs):
        """
        Args:
            kwargs (:obj:`dict`, optional): Command-line arguments to override
                (used when invoked from python):
                `input` (str) - Path to .conll input file.
                `outdir` (str) - Path to output directory.
                `model` (str) - Path to .model file to load the model from.
                `eval` (bool) - Specifies whether to generate an evaluation file.

        Returns:
            A 3-tuple consisting of model options, working and output directories.
        """
        opt_parser = parser_utils.get_option_parser(kwargs)
        opt_parser.add_option("--input", dest="conll_input",
                              help="Annotated CONLL input file",
                              metavar="FILE",
                              default=kwargs.get('input'))
        opt_parser.add_option("--eval", action="store_true", dest="evalFlag",
                              default=kwargs.get('eval', False))
        options = opt_parser.parse_args()[0]
        is_conllu = os.path.splitext(options.conll_input.lower())[1] == '.conllu'
        write_path = os.path.join(
            options.output, 'inference_res.conll' if not is_conllu else 'inference_res.conllu')
        root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        return options, root_dir, write_path

    @staticmethod
    def load_model(path):
        """Load and initialize a MSTParserLSTM model from a .model file."""
        params_path = os.path.join(pathlib.Path(os.path.abspath(path)).parent, 'params.pickle')
        words, w2i, pos, rels, stored_opt = pickle.load(io.open(params_path, 'rb'))
        print('Initializing BIST MSTParserLSTM...')
        parser_model = MSTParserLSTM(words, w2i, pos, rels, stored_opt)
        print('Done')
        parser_model.load(path)
        return parser_model
