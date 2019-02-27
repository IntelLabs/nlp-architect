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

import json
import logging
import sys

from gensim.models import FastText, Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
from gensim import utils
import nltk
from nltk.corpus import conll2000
from six import iteritems

logger = logging.getLogger(__name__)


# pylint: disable-msg=too-many-instance-attributes
class NP2vec:
    """
    Initialize the np2vec model, train it, save it and load it.
    """

    def is_marked(self, s):
        """
        Check if a string is marked.

        Args:
            s (str): string to check
        """
        return len(s) > 0 and s[-1] == self.mark_char

    # pylint: disable-msg=too-many-arguments
    # pylint: disable-msg=too-many-locals
    # pylint: disable-msg=too-many-branches
    def __init__(  # noqa: C901
            self,
            corpus,
            corpus_format='txt',
            mark_char='_',
            word_embedding_type='word2vec',
            sg=0,
            size=100,
            window=10,
            alpha=0.025,
            min_alpha=0.0001,
            min_count=5,
            sample=1e-5,
            workers=20,
            hs=0,
            negative=25,
            cbow_mean=1,
            iterations=15,
            min_n=3,
            max_n=6,
            word_ngrams=1,
            prune_non_np=True):
        """
        Initialize np2vec model and train it.

        Args:
          corpus (str): path to the corpus.
          corpus_format (str {json,txt,conll2000}): format of the input marked corpus; txt and json
          formats are supported. For json format, the file should contain an iterable of
          sentences. Each sentence is a list of terms (unicode strings) that will be used for
          training.
          mark_char (char): special character that marks NP's suffix.
          word_embedding_type (str {word2vec,fasttext}): word embedding model type; word2vec and
          fasttext are supported.
          np2vec_model_file (str): path to the file where the trained np2vec model has to be
          stored.
          binary (bool): boolean indicating whether the model is stored in binary format; if
          word_embedding_type is fasttext and word_ngrams is 1, binary should be set to True.
          sg (int {0,1}): model training hyperparameter, skip-gram. Defines the training
          algorithm. If 1, CBOW is used,otherwise, skip-gram is employed.
          size (int): model training hyperparameter, size of the feature vectors.
          window (int): model training hyperparameter, maximum distance between the current and
          predicted word within a sentence.
          alpha (float): model training hyperparameter. The initial learning rate.
          min_alpha (float): model training hyperparameter. Learning rate will linearly drop to
          `min_alpha` as training progresses.
          min_count (int): model training hyperparameter, ignore all words with total frequency
          lower than this.
          sample (float): model training hyperparameter, threshold for configuring which
          higher-frequency words are randomly downsampled, useful range is (0, 1e-5)
          workers (int): model training hyperparameter, number of worker threads.
          hs (int {0,1}): model training hyperparameter, hierarchical softmax. If set to 1,
          hierarchical softmax will be used for model training. If set to 0, and `negative` is non-
                        zero, negative sampling will be used.
          negative (int): model training hyperparameter, negative sampling. If > 0, negative
          sampling will be used, the int for negative specifies how many "noise words" should be
          drawn (usually between 5-20). If set to 0, no negative sampling is used.
          cbow_mean (int {0,1}): model training hyperparameter. If 0, use the sum of the context
          word vectors. If 1, use the mean, only applies when cbow is used.
          iterations (int): model training hyperparameter, number of iterations.
          min_n (int): fasttext training hyperparameter. Min length of char ngrams to be used
          for training word representations.
          max_n (int): fasttext training hyperparameter. Max length of char ngrams to be used for
          training word representations. Set `max_n` to be lesser than `min_n` to avoid char
          ngrams being used.
          word_ngrams (int {0,1}): fasttext training hyperparameter. If 1, uses enrich word
          vectors with subword (ngrams) information. If 0, this is equivalent to word2vec training.
          prune_non_np (bool): indicates whether to prune non-NP's after training process.

        """

        self.mark_char = mark_char
        self.word_embedding_type = word_embedding_type
        self.sg = sg
        self.size = size
        self.window = window
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.min_count = min_count
        self.sample = sample
        self.workers = workers
        self.hs = hs
        self.negative = negative
        self.cbow_mean = cbow_mean
        self.iter = iterations
        self.min_n = min_n
        self.max_n = max_n
        self.word_ngrams = word_ngrams
        self.prune_non_np = prune_non_np

        if corpus_format == 'txt':
            self._sentences = LineSentence(corpus)
        elif corpus_format == 'json':
            with open(corpus) as json_data:
                self._sentences = json.load(json_data)
        # pylint: disable-msg=too-many-nested-blocks
        elif corpus_format == 'conll2000':
            try:
                self._sentences = list()
                for chunked_sent in conll2000.chunked_sents(corpus):
                    tokens = list()
                    for chunk in chunked_sent:
                        # pylint: disable-msg=protected-access
                        if hasattr(chunk, '_label') and chunk._label == 'NP':
                            s = ''
                            for w in chunk:
                                s += w[0] + self.mark_char
                            tokens.append(s)
                        else:
                            if isinstance(chunk, nltk.Tree):
                                for w in chunk:
                                    tokens.append(w[0])
                            else:
                                tokens.append(chunk[0])
                        self._sentences.append(tokens)
            # pylint: disable-msg=broad-except
            except Exception:
                print('Conll2000 dataset is missing. See downloading details in the '
                      'README file')
        else:
            logger.error('invalid corpus format: %s', corpus_format)
            sys.exit(0)

        if word_embedding_type == 'fasttext' and word_ngrams == 1:
            # remove the marking character at the end for subword fasttext model training
            self._sentences = [[w[:-1] if self.is_marked(w) else w for w in sentence]
                               for sentence in self._sentences]

        logger.info('training np2vec model')
        self._train()

    def _train(self):
        """
        Train the np2vec model.
        """
        if self.word_embedding_type == 'word2vec':
            self.model = Word2Vec(
                self._sentences,
                sg=self.sg,
                size=self.size,
                window=self.window,
                alpha=self.alpha,
                min_alpha=self.min_alpha,
                min_count=self.min_count,
                sample=self.sample,
                workers=self.workers,
                hs=self.hs,
                negative=self.negative,
                cbow_mean=self.cbow_mean,
                iter=self.iter)

        elif self.word_embedding_type == 'fasttext':
            self.model = FastText(
                self._sentences,
                sg=self.sg,
                size=self.size,
                window=self.window,
                alpha=self.alpha,
                min_alpha=self.min_alpha,
                min_count=self.min_count,
                sample=self.sample,
                workers=self.workers,
                hs=self.hs,
                negative=self.negative,
                cbow_mean=self.cbow_mean,
                iter=self.iter,
                min_n=self.min_n,
                max_n=self.max_n,
                word_ngrams=self.word_ngrams)
        else:
            logger.error('invalid word embedding type: %s', self.word_embedding_type)
            sys.exit(0)

    def save(self, np2vec_model_file='np2vec.model', binary=False, word2vec_format=True):
        """
        Save the np2vec model.

        Args:
            np2vec_model_file (str): the file containing the np2vec model to load
            binary (bool): boolean indicating whether the np2vec model to load is in binary format
            word2vec_format(bool): boolean indicating whether to save the model in original
            word2vec format.
        """
        if self.word_embedding_type == 'fasttext' and self.word_ngrams == 1:
            if not binary:
                logger.error(
                    "if word_embedding_type is fasttext and word_ngrams is 1, "
                    "binary should be set to True.")
                sys.exit(0)
            # not relevant to prune fasttext subword model
            self.model.save(np2vec_model_file)
        else:
            # prune non NP terms
            if self.prune_non_np:
                logger.info('pruning np2vec model')
                total_vec = 0
                vector_size = self.model.vector_size
                for word in self.model.wv.vocab.keys():
                    if self.is_marked(word) and len(word) > 1:
                        total_vec += 1
                logger.info(
                    "storing %sx%s projection weights for NP's into %s",
                    total_vec, vector_size, np2vec_model_file)
                with utils.smart_open(np2vec_model_file, 'wb') as fout:
                    fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
                    # store NP vectors in sorted order: most frequent NP's at the top
                    for word, vocab in sorted(
                            iteritems(
                                self.model.wv.vocab), key=lambda item: -item[1].count):
                        if self.is_marked(word) and len(word) > 1:  # discard empty marked np's
                            embedding_vec = self.model.wv.syn0[vocab.index]
                            if binary:
                                fout.write(
                                    utils.to_utf8(word) + b" " + embedding_vec.tostring())
                            else:
                                fout.write(
                                    utils.to_utf8(
                                        "%s %s\n" %
                                        (word, ' '.join(
                                            "%f" %
                                            val for val in embedding_vec))))
                if not word2vec_format:
                    # pylint: disable=attribute-defined-outside-init
                    self.model = KeyedVectors.load_word2vec_format(np2vec_model_file,
                                                                   binary=binary)
            if not word2vec_format:
                self.model.save(np2vec_model_file)

    @classmethod
    def load(cls, np2vec_model_file, binary=False, word_ngrams=0, word2vec_format=True):
        """
        Load the np2vec model.

        Args:
            np2vec_model_file (str): the file containing the np2vec model to load
            binary (bool): boolean indicating whether the np2vec model to load is in binary format
            word_ngrams (int {1,0}): If 1, np2vec model to load uses word vectors with subword (
            ngrams) information.
            word2vec_format(bool): boolean indicating whether the model to load has been stored in
            original word2vec format.

        Returns:
            np2vec model to load
        """
        if word_ngrams == 0:
            if word2vec_format:
                return KeyedVectors.load_word2vec_format(np2vec_model_file, binary=binary)
            return KeyedVectors.load(np2vec_model_file, mmap='r')
        if word_ngrams == 1:
            return FastText.load(np2vec_model_file)
        logger.error('invalid value for \'word_ngrams\'')
        return None
