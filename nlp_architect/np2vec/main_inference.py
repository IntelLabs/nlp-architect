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

import logging
import sys
from configargparse import ArgumentParser

from nlp_architect.np2vec import NP2vec

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    arg_parser = ArgumentParser(__doc__)
    arg_parser.add_argument(
        '--np2vec_model_file',
        default='sample_np2vec.model',
        help='path to the file with the np2vec model to load.')
    arg_parser.add_argument(
        '--binary',
        help='boolean indicating whether the model to load has been stored in binary '
        'format.',
        action='store_true')
    arg_parser.add_argument(
        '--word_ngrams',
        default=0,
        type=int,
        choices=[0, 1],
        help='If 0, the model to load stores word information. If 1, the model to load stores '
        'subword (ngrams) information; note that subword information is relevant only to '
        'fasttext models.')

    args = arg_parser.parse_args()

    np2vec_model = NP2vec.load(
        args.np2vec_model_file,
        binary=args.binary,
        word_ngrams=args.word_ngrams)

    print("word vector for the NP \'Intel\':", np2vec_model['Intel_'])
    if args.word_ngrams == 1:
        print("word vector for the NP \'Intel_Organization\':",
              np2vec_model['Intel_Organization_'])
