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
import json

from configargparse import ArgumentParser

from nlp_architect.models.np2vec import NP2vec
from nlp_architect.utils.io import validate_existing_filepath, check_size, load_json_file

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class SetExpand():
    """
        Set expansion module, given a trained np2vec model.
    """

    def __init__(self, np2vec_model_file, binary=False, word_ngrams=False):
        """
        Load the np2vec model for set expansion.

        Args:
            np2vec_model_file (str): the file containing the np2vec model to load
            binary (bool): boolean indicating whether the np2vec model to load is in binary format
            word_ngrams (int {1,0}): If 1, np2vec model to load uses word vectors with subword (
            ngrams) information.

        Returns:
            np2vec model to load
        """
        # load grouping info
        logger.info('loading grouping data')
        self.id2rep = load_json_file('id2rep')
        self.np2id = load_json_file('np2id')
        self.id2group = load_json_file('id2group')
        logger.info('loadind model...')
        self.np2vec_model = NP2vec.load(np2vec_model_file, binary=binary, word_ngrams=word_ngrams)
        # extract the first term of the model in order to get the marking character
        logger.info('compute L2 norm')
        first_term = next(iter(self.np2vec_model.vocab.keys()))
        self.mark_char = first_term[-1]
        # Precompute L2-normalized vectors.
        self.np2vec_model.init_sims()
        logger.info('done init')


    def __term2id(self, term):
        """
        Given an term, return its id.

        Args:
            term(str): term (noun phrase)

        Returns:
            its id
        """
        return term.replace(' ', self.mark_char) + self.mark_char

    def __id2term(self, id):
        """
        Given the id of a noun phrase, return the noun phrase string.

        Args:
            id(str): id

         Returns:
            term (noun phrase)
        """
        return id.replace(self.mark_char, ' ')[:-1]

    def get_vocab(self):
        """
        Return the vocabulary as the list of terms.

        Returns:
            the list of terms.
        """
        return [self.__id2term(id) for id in self.np2vec_model.vocab]

    def in_vocab(self, term):
        norm = None
        logger.info("check is in vocab: " + term)
        if term in self.np2id.keys():
            norm = self.np2id[term]
        if norm is None or norm not in self.id2rep or self.__term2id(
                    self.id2rep[norm]) not in self.np2vec_model.vocab:
            return False
        return True

    def get_group(self, term):
        logger.info("check is in vocab: " + term)
        group = []
        if term in self.np2id:
            id = self.np2id[term]
            if id in self.id2group:
                group = self.id2group[id]
        return group

    def expand(self, seed, topn=500):
        """
        Given a seed of terms, return the expanded set of terms.

        Args:
            seed: seed terms
            topn: maximal number of expanded terms to return

        Returns:
            up to topn expanded terms and their probabilities
        """
        seed_ids = list()
        norm = None
        for np in seed:
            if np in self.np2id.keys():
                norm = self.np2id[np]
            if norm is None or norm not in self.id2rep or self.__term2id(
                    self.id2rep[norm]) not in self.np2vec_model.vocab:
                logger.warning("The term: '" + np + "' is out-of-vocabulary.")
            else:
                id = self.__term2id(self.id2rep[norm])
                seed_ids.append(id)
        if len(seed_ids) > 0:
            logger.info("seed_ids=" + str(seed_ids))
            res_id = self.np2vec_model.most_similar(seed_ids, topn=topn)
            res = list()
            for r in res_id:
                res.append((self.__id2term(r[0]), r[1]))
            return res
        else:
            logger.info("All the seed terms are out-of-vocabulary.")
        return None


if __name__ == "__main__":
    arg_parser = ArgumentParser(__doc__)
    arg_parser.add_argument(
        '--np2vec_model_file',
        help='path to the file with the np2vec model to load.',
        type=validate_existing_filepath)
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
    arg_parser.add_argument('--seed', type=str, action=check_size(min_size=1),\
    help='comma-separated seed terms')
    arg_parser.add_argument('--topn', default=500, type=int, action=check_size(min_size=1),
                            help='maximal number of expanded terms to return')

    args = arg_parser.parse_args()
    se = SetExpand(np2vec_model_file=args.np2vec_model_file, binary=args.binary,
                   word_ngrams=args.word_ngrams)
    exp = se.expand(args.seed, args.topn)
    logger.info('Expanded results:')
    logger.info(exp)
