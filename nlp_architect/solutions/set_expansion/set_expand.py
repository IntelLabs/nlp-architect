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
from os import path

from argparse import ArgumentParser

from nlp_architect.models.np2vec import NP2vec
from nlp_architect.utils.io import validate_existing_filepath, check_size, load_json_file

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class SetExpand(object):
    """
        Set expansion module, given a trained np2vec model.
    """

    def __init__(self, np2vec_model_file, binary=False, word_ngrams=False, grouping=False,
                 light_grouping=False, grouping_map_dir=None):
        """
        Load the np2vec model for set expansion.

        Args:
            np2vec_model_file (str): the file containing the np2vec model to load
            binary (bool): boolean indicating whether the np2vec model to load is in binary format
            word_ngrams (int {1,0}): If 1, np2vec model to load uses word vectors with subword (
            ngrams) information.
            light_grouping (bool): boolean indicating whether to load all maps for grouping.
            grouping_map_dir (str): path to the directory containing maps for grouping.
        Returns:
            np2vec model to load
        """
        self.grouping = grouping
        if grouping:
            # load grouping info
            logger.info('loading grouping data')
            if not grouping_map_dir:
                grouping_map_dir = path.dirname(np2vec_model_file)
            self.np2id = load_json_file(path.join(grouping_map_dir, 'np2id'))
            if not light_grouping:
                self.id2rep = load_json_file(path.join(grouping_map_dir, 'id2rep'))
                self.id2group = load_json_file(path.join(grouping_map_dir, 'id2group'))
        logger.info('loadind model...')
        self.np2vec_model = NP2vec.load(np2vec_model_file, binary=binary, word_ngrams=word_ngrams)
        # extract the first term of the model in order to get the marking character
        logger.info('compute L2 norm')
        first_term = next(iter(self.np2vec_model.vocab.keys()))
        self.mark_char = first_term[-1]
        # Precompute L2-normalized vectors.
        self.np2vec_model.init_sims()
        logger.info('done init')

    def term2id(self, term, suffix=True):
        """
        Given an term, return its id.

        Args:
            term(str): term (noun phrase)

        Returns:
            its id (if is part of the model)
        """
        if self.grouping:
            if term not in self.np2id.keys():
                return None
            term = self.np2id[term]
        id = term.replace(' ', self.mark_char)
        if suffix:
            id += self.mark_char
        if id not in self.np2vec_model.vocab:
            return None
        return id

    def __id2term(self, id):
        """
        Given the id of a noun phrase, return the noun phrase string.

        Args:
            id(str): id

         Returns:
            term (noun phrase)
        """
        norm = id.replace(self.mark_char, ' ')[:-1]
        if self.grouping:
            if norm in self.id2rep:
                return self.id2rep[norm]
            logger.warning("id:#%s#, norm:#%s# is not in id2rep")
            return ""
        return norm

    def get_vocab(self):
        """
        Return the vocabulary as the list of terms.

        Returns:
            the list of terms.
        """
        vocab = []
        for id in self.np2vec_model.vocab:
            term = self.__id2term(id)
            if term is not None:
                vocab.append(term)
        return vocab

    def in_vocab(self, term):
        id = self.term2id(term)
        if id is None:
            return False
        return True

    def get_group(self, term):
        logger.info("get group of: %s", term)
        group = []
        if term in self.np2id:
            id = self.np2id[term]
            group = self.id2group[id]
        return group

    def similarity(self, terms, seed, threshold):

        similar = []
        seed_id = self.get_seed_id(seed)
        for term in terms:
            term_id = self.term2id(term)
            if term_id is not None:
                if self.seed2term_similarity(seed_id, [term_id]) > threshold:
                    similar.append(term)
            else:
                logger.info("term: %s is not in vocab", term)
        return similar

    # pylint: disable-msg=too-many-branches
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
        upper = True
        lower = True
        for np in seed:
            np = np.strip()
            if not self.grouping and (upper or lower):
                # case feature is relevant only if we don't have grouping
                if np[0].islower():
                    upper = False
                else:
                    lower = False
            id = self.term2id(np)
            if id is not None:
                seed_ids.append(id)
            else:
                logger.warning("The term: '%s' is out-of-vocabulary.", np)
        if len(seed_ids) > 0:
            if not self.grouping and (upper or lower):
                res_id = self.np2vec_model.most_similar(seed_ids, topn=2 * topn)
            else:
                res_id = self.np2vec_model.most_similar(seed_ids, topn=topn)
            res = list()
            for r in res_id:
                if len(res) == topn:
                    break
                # pylint: disable=R0916
                if self.grouping or (not lower and not upper) or (upper and r[0][0].isupper()) or \
                        (lower and r[0][0].islower()):
                    term = self.__id2term(r[0])
                    if term is not None:
                        res.append((self.__id2term(r[0]), r[1]))
            ret_val = res
        else:
            logger.info("All the seed terms are out-of-vocabulary.")
            ret_val = None
        return ret_val

    def get_seed_id(self, seed):
        seed_ids = list()
        for np in seed:
            np = np.strip()
            id = self.term2id(np)
            if id is not None:
                seed_ids.append(id)
            else:
                logger.warning("The term: '%s' is out-of-vocabulary.", np)
        return seed_ids

    def term2term_similarity(self, term_id_1, term_id_2):
        """
        Compute cosine similarity between two term id's.
        Args:
            term_id_1: first term id
            term_id_2: second term id

        Returns:
            Similarity between the first and second term id's

        """
        logger.info('calculate similarity for: %s , %s', term_id_1, term_id_2)
        res = self.np2vec_model.similarity(term_id_1, term_id_2)
        logger.info("similarity result: %s", str(res))
        return res

    def seed2term_similarity(self, seed_id, term_id):
        """
        Compute cosine similarity between a seed terms and a term.
        Args:
            seed_id: seed term id's
            term_id: the term id

        Returns:
            Similarity between the seed terms and the term

        """
        logger.info('calculate similarity for: %s , %s', str(seed_id), term_id)
        res = self.np2vec_model.n_similarity(seed_id, list(term_id))
        logger.info("similarity result: %s", str(res))
        return res


if __name__ == "__main__":
    arg_parser = ArgumentParser(__doc__)
    arg_parser.add_argument(
        '--np2vec_model_file',
        help='path to the file with the np2vec model to load.',
        type=validate_existing_filepath)
    arg_parser.add_argument(
        '--binary',
        help='boolean indicating whether the model to load has been stored in binary format.',
        action='store_true')
    arg_parser.add_argument(
        '--word_ngrams',
        default=0,
        type=int,
        choices=[0, 1],
        help='If 0, the model to load stores word information. If 1, the model to load stores '
             'subword (ngrams) information; note that subword information is relevant only to '
             'fasttext models.')
    arg_parser.add_argument(
        '--topn',
        default=500,
        type=int,
        action=check_size(min_size=1),
        help='maximal number of expanded terms to return')
    arg_parser.add_argument(
        '--grouping',
        action='store_true',
        default=False,
        help='grouping mode')

    args = arg_parser.parse_args()

    se = SetExpand(np2vec_model_file=args.np2vec_model_file, binary=args.binary,
                   word_ngrams=args.word_ngrams, grouping=args.grouping)
    enter_seed_str = 'Enter the seed (comma-separated seed terms):'
    logger.info(enter_seed_str)
    for seed_str in sys.stdin:
        seed_list = seed_str.strip().split(',')
        exp = se.expand(seed_list, args.topn)
        logger.info('Expanded results:')
        logger.info(exp)
        logger.info(enter_seed_str)
