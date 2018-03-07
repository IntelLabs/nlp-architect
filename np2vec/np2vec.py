import json
import logging
import sys

from gensim.models import FastText, Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
from gensim import utils
from six import iteritems

logger = logging.getLogger(__name__)


class NP2vec:

    def is_marked(self, s):
        return len(s) > 0 and s[-1] == self.mark_char

    def __init__(
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
            iter=15,
            min_n=3,
            max_n=6,
            word_ngrams=1):
        """
        Initialize the np2vec model and train it.
        """
        self.mark_char = mark_char
        self.word_embedding_type = word_embedding_type
        self.word_ngrams = word_ngrams
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
        self.iter = iter
        self.min_n = min_n
        self.max_n = max_n
        self.word_ngrams = word_ngrams

        if corpus_format == 'txt':
            self._sentences = LineSentence(corpus)
        elif corpus_format == 'json':
            with open(corpus) as json_data:
                self._sentences = json.load(json_data)
        else:
            logger.error('invalid corpus format: ' + corpus_format)
            sys.exit(0)

        if word_embedding_type == 'fasttext' and word_ngrams == 1:
            # remove the '_' at the end for subword fasttext model training
            for i, sentence in enumerate(self._sentences):
                self._sentences[i] = [
                    w[:-1] if self.is_marked(w) else w for w in sentence]

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
                iter=iter,
                min_n=self.min_n,
                max_n=self.max_n,
                word_ngrams=self.word_ngrams)
        else:
            logger.error(
                'invalid word embedding type: ' +
                self.word_embedding_type)
            sys.exit(0)

    def save(self, np2vec_model_file='np2vec.model', binary=False):
        """
        Save the np2vec model.
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
            logger.info('pruning np2vec model')
            total_vec = 0
            vector_size = self.model.vector_size
            for word in self.model.wv.vocab.keys():
                if self.is_marked(word):
                    total_vec += 1
            logger.info(
                "storing %sx%s projection weights into %s" %
                (total_vec, vector_size, np2vec_model_file))
            with utils.smart_open(np2vec_model_file, 'wb') as fout:
                fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
                # store in sorted order: most frequent words at the top
                for word, vocab in sorted(
                    iteritems(
                        self.model.wv.vocab), key=lambda item: -item[1].count):
                    if self.is_marked(word):
                        row = self.model.wv.syn0[vocab.index]
                        if binary:
                            fout.write(
                                utils.to_utf8(word) + b" " + row.tostring())
                        else:
                            fout.write(
                                utils.to_utf8(
                                    "%s %s\n" %
                                    (word, ' '.join(
                                        "%f" %
                                        val for val in row))))

    @classmethod
    def load(cls, np2vec_model_file, binary=False, word_ngrams=0):
        """
        Load the np2vec model
        """
        if word_ngrams == 0:
            return KeyedVectors.load_word2vec_format(
                np2vec_model_file, binary=binary)
        elif word_ngrams == 1:
            return FastText.load(np2vec_model_file)
        else:
            logger.error('invalid value for \'word_ngrams\'')
