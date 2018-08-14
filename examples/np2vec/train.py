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
import argparse
import logging
import sys

from nlp_architect.models.np2vec import NP2vec
from nlp_architect.utils.io import check_size, validate_existing_filepath

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--corpus',
        default='train.txt',
        type=str,
        action=check_size(min_size=1),
        help='path to the corpus. By default, '
             'it is the training set of CONLL2000 shared task dataset.')
    arg_parser.add_argument(
        '--corpus_format',
        default='conll2000',
        type=str,
        choices=[
            'json',
            'txt',
            'conll2000'],
        help='format of the input marked corpus; txt, conll2000 and json formats are supported. '
        'For json format, the file should contain an iterable of sentences. '
        'Each sentence is a list of terms (unicode strings) that will be used for training.')
    arg_parser.add_argument(
        '--mark_char',
        default='_',
        type=str,
        action=check_size(1, 2),
        help='special character that marks word separator and NP suffix.')
    arg_parser.add_argument(
        '--word_embedding_type',
        default='word2vec',
        type=str,
        choices=[
            'word2vec',
            'fasttext'],
        help='word embedding model type; word2vec and fasttext are supported.')
    arg_parser.add_argument(
        '--np2vec_model_file',
        default='conll2000.train.model',
        type=str,
        action=check_size(min_size=1),
        help='path to the file where the trained np2vec model has to be stored.')
    arg_parser.add_argument(
        '--binary',
        help='boolean indicating whether the model is stored in binary format; if '
        'word_embedding_type is fasttext and word_ngrams is 1, '
        'binary should be set to True.',
        action='store_true')

    # word2vec and fasttext common hyperparameters
    arg_parser.add_argument(
        '--sg',
        default=0,
        type=int,
        choices=[
            0,
            1],
        help='model training hyperparameter, skip-gram. Defines the training algorithm. If 1, '
        'CBOW is used, otherwise, skip-gram is employed.')
    arg_parser.add_argument(
        '--size',
        default=100,
        type=int,
        action=check_size(min_size=1),
        help='model training hyperparameter, size of the feature vectors.')
    arg_parser.add_argument(
        '--window',
        default=10,
        type=int,
        action=check_size(min_size=1),
        help='model training hyperparameter, maximum distance '
        'between the current and predicted word within a '
        'sentence.')
    arg_parser.add_argument(
        '--alpha',
        default=0.025,
        type=float,
        action=check_size(min_size=0),
        help='model training hyperparameter. The initial learning rate.')
    arg_parser.add_argument(
        '--min_alpha',
        default=0.0001,
        type=float,
        action=check_size(min_size=0),
        help='model training hyperparameter. Learning rate will linearly drop to `min_alpha` as '
        'training progresses.')
    arg_parser.add_argument(
        '--min_count',
        default=5,
        type=int,
        action=check_size(min_size=0),
        help='model training hyperparameter, ignore all words '
        'with total frequency lower than this.')
    arg_parser.add_argument(
        '--sample',
        default=1e-5,
        type=float,
        action=check_size(min_size=0),
        help='model training hyperparameter, threshold for '
        'configuring which higher-frequency words are '
        'randomly downsampled, useful range is (0, 1e-5)')
    arg_parser.add_argument(
        '--workers',
        default=20,
        type=int,
        action=check_size(min_size=1),
        help='model training hyperparameter, number of worker threads.')
    arg_parser.add_argument(
        '--hs',
        default=0,
        type=int,
        choices=[
            0,
            1],
        help='model training hyperparameter, hierarchical softmax. If set to 1, hierarchical '
        'softmax will be used for model training. '
        'If set to 0, and `negative` is non-zero, negative sampling will be used.')
    arg_parser.add_argument(
        '--negative',
        default=25,
        type=int,
        action=check_size(min_size=0),
        help='model training hyperparameter, negative sampling. If > 0, negative sampling will be '
        'used, the int for negative specifies how many \"noise words\" should be drawn ('
        'usually between 5-20). If set to 0, no negative sampling is used.')
    arg_parser.add_argument(
        '--cbow_mean',
        default=1,
        type=int,
        choices=[
            0,
            1],
        help='model training hyperparameter.  If 0, use the sum of the context word vectors. '
             'If 1, use the mean, only applies when cbow is used.')
    arg_parser.add_argument(
        '--iter',
        default=15,
        type=int,
        action=check_size(min_size=1),
        help='model training hyperparameter, number of iterations.')

    # fasttext hyperparameters
    arg_parser.add_argument(
        '--min_n',
        default=3,
        type=int,
        action=check_size(min_size=1),
        help='fasttext training hyperparameter. Min length of char ngrams to be used for training '
        'word representations.')
    arg_parser.add_argument(
        '--max_n',
        default=6,
        type=int,
        action=check_size(min_size=0),
        help='fasttext training hyperparameter. Max length of char ngrams to be used for training '
        'word representations. '
        'Set `max_n` to be lesser than `min_n` to avoid char ngrams being used.')
    arg_parser.add_argument(
        '--word_ngrams',
        default=1,
        type=int,
        choices=[
            0,
            1],
        help='fasttext training hyperparameter. If 1, uses enrich word vectors with subword ('
        'ngrams) information. If 0, this is equivalent to word2vec training.')

    args = arg_parser.parse_args()

    if args.corpus_format != 'conll2000':
        validate_existing_filepath(args.corpus)

    np2vec = NP2vec(
        args.corpus,
        args.corpus_format,
        args.mark_char,
        args.word_embedding_type,
        args.sg,
        args.size,
        args.window,
        args.alpha,
        args.min_alpha,
        args.min_count,
        args.sample,
        args.workers,
        args.hs,
        args.negative,
        args.cbow_mean,
        args.iter,
        args.min_n,
        args.max_n,
        args.word_ngrams)

    np2vec.save(args.np2vec_model_file, args.binary)
