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

from __future__ import division, print_function, unicode_literals, absolute_import

import sys

import nltk
import numpy as np
from neon.data.text_preprocessing import pad_sentences
from nlp_architect.contrib.neon.text_iterators import TaggedTextSequence, MultiSequenceDataIterator
from nlp_architect.utils.embedding import load_word_embeddings
from nlp_architect.utils.generic import get_paddedXY_sequence, license_prompt
from nltk.corpus import conll2000


class CONLL2000(object):
    """
    CONLL 2000 chunking task data set (Neon)

    Arguments:
        sentence_length (int): number of time steps to embed the data.
        vocab_size (int): max size of vocabulary.
        use_pos (boolean, optional): Yield POS tag features.
        use_chars (boolean, optional): Yield Char RNN features.
        use_w2v (boolean, optional): Use W2V as input features.
        w2v_path (str, optional): W2V model path
    """

    def __init__(self, sentence_length=50, vocab_size=20000,
                 use_pos=False,
                 use_chars=False,
                 chars_len=20,
                 use_w2v=False,
                 w2v_path=None):
        self.sentence_length = sentence_length
        self.vocab_size = vocab_size
        self.use_pos = use_pos
        self.use_chars = use_chars
        self.chars_len = chars_len
        self.use_w2v = use_w2v
        self.w2v_path = w2v_path
        self.vocabs = {}
        self._data_dict = {}

    @staticmethod
    def load_data():
        try:
            train_set = conll2000.chunked_sents('train.txt')
            test_set = conll2000.chunked_sents('test.txt')
        except Exception:
            if license_prompt('CONLL2000 data set',
                              'http://www.nltk.org/nltk_data/',
                              'Apache 2.0',
                              'https://github.com/nltk/nltk/blob/develop/LICENSE.txt') is False:
                sys.exit(0)
            nltk.download('conll2000')
            train_set = conll2000.chunked_sents('train.txt')
            test_set = conll2000.chunked_sents('test.txt')
        train_data = [list(zip(*nltk.chunk.tree2conlltags(sent))) for sent in train_set]
        test_data = [list(zip(*nltk.chunk.tree2conlltags(sent))) for sent in test_set]
        return train_data, test_data

    def create_char_features(self, sentences, sentence_length, word_length):
        char_dict = {}
        char_id = 3
        new_sentences = []
        for s in sentences:
            char_sents = []
            for w in s:
                char_vector = []
                for c in w:
                    char_int = char_dict.get(c, None)
                    if char_int is None:
                        char_dict[c] = char_id
                        char_int = char_id
                        char_id += 1
                    char_vector.append(char_int)
                char_vector = [1] + char_vector + [2]
                char_sents.append(char_vector)
            char_sents = pad_sentences(char_sents, sentence_length=word_length)
            if sentence_length - char_sents.shape[0] < 0:
                char_sents = char_sents[:sentence_length]
            else:
                padding = np.zeros(
                    (sentence_length - char_sents.shape[0], word_length))
                char_sents = np.vstack((padding, char_sents))
            new_sentences.append(char_sents)
        char_sentences = np.asarray(new_sentences)
        self.vocabs.update({'char_rnn': char_dict})
        return char_sentences

    @property
    def train_iter(self):
        if self._data_dict.get('train', None) is None:
            self.gen_iterators()
        return self._data_dict.get('train')

    @property
    def test_iter(self):
        if self._data_dict.get('test', None) is None:
            self.gen_iterators()
        return self._data_dict.get('test')

    def gen_iterators(self):
        train_set, test_set = self.load_data()
        num_train_samples = len(train_set)

        sents = list(zip(*train_set))[0] + list(zip(*test_set))[0]
        X, X_vocab = self._sentences_to_ints(sents, lowercase=False)
        self.vocabs.update({'token': X_vocab})

        y = list(zip(*train_set))[2] + list(zip(*test_set))[2]
        y, y_vocab = self._sentences_to_ints(y, lowercase=False)
        self.y_vocab = y_vocab
        X, y = get_paddedXY_sequence(
            X, y, sentence_length=self.sentence_length, shuffle=False)

        self._data_dict = {}
        self.y_size = len(y_vocab) + 1
        train_iters = []
        test_iters = []

        if self.use_w2v:
            w2v_dict, emb_size = load_word_embeddings(self.w2v_path)
            self.emb_size = emb_size
            x_vocab_is = {i: s for s, i in X_vocab.items()}
            X_w2v = []
            for xs in X:
                _xs = []
                for w in xs:
                    if 0 <= w <= 2:
                        _xs.append(np.zeros(emb_size))
                    else:
                        word = x_vocab_is[w - 3]
                        vec = w2v_dict.get(word.lower())
                        if vec is not None:
                            _xs.append(vec)
                        else:
                            _xs.append(np.zeros(emb_size))
                X_w2v.append(_xs)
            X_w2v = np.asarray(X_w2v)
            train_iters.append(TaggedTextSequence(self.sentence_length,
                                                  x=X_w2v[:num_train_samples],
                                                  y=y[:num_train_samples],
                                                  num_classes=self.y_size,
                                                  vec_input=True))
            test_iters.append(TaggedTextSequence(self.sentence_length,
                                                 x=X_w2v[num_train_samples:],
                                                 y=y[num_train_samples:],
                                                 num_classes=self.y_size,
                                                 vec_input=True))
        else:
            train_iters.append(TaggedTextSequence(self.sentence_length,
                                                  x=X[:num_train_samples],
                                                  y=y[:num_train_samples],
                                                  num_classes=self.y_size))
            test_iters.append(TaggedTextSequence(self.sentence_length,
                                                 x=X[num_train_samples:],
                                                 y=y[num_train_samples:],
                                                 num_classes=self.y_size))

        if self.use_pos:
            pos_sents = list(zip(*train_set))[1] + list(zip(*test_set))[1]
            X_pos, X_pos_vocab = self._sentences_to_ints(pos_sents)
            self.vocabs.update({'pos': X_pos_vocab})
            X_pos, _ = get_paddedXY_sequence(X_pos, y, sentence_length=self.sentence_length,
                                             shuffle=False)
            train_iters.append(TaggedTextSequence(steps=self.sentence_length,
                                                  x=X_pos[:num_train_samples]))
            test_iters.append(TaggedTextSequence(steps=self.sentence_length,
                                                 x=X_pos[num_train_samples:]))

        if self.use_chars:
            char_sentences = self.create_char_features(
                sents, self.sentence_length, self.chars_len)
            char_sentences = char_sentences.reshape(
                -1, self.sentence_length * self.chars_len)
            char_train = char_sentences[:num_train_samples]
            char_test = char_sentences[num_train_samples:]
            train_iters.append(TaggedTextSequence(steps=self.chars_len * self.sentence_length,
                                                  x=char_train))
            test_iters.append(TaggedTextSequence(steps=self.chars_len * self.sentence_length,
                                                 x=char_test))

        if len(train_iters) > 1:
            self._data_dict['train'] = MultiSequenceDataIterator(train_iters)
            self._data_dict['test'] = MultiSequenceDataIterator(test_iters)
        else:
            self._data_dict['train'] = train_iters[0]
            self._data_dict['test'] = test_iters[0]
        return self._data_dict

    @staticmethod
    def _sentences_to_ints(texts, lowercase=True):
        """
        convert text sentences into int id sequences. Word ids are sorted
        by frequency of appearance.
        return int sequences and vocabulary.
        """
        w_dict = {}
        for sen in texts:
            for w in sen:
                if lowercase:
                    w = w.lower()
                w_dict.update({w: w_dict.get(w, 0) + 1})
        int_to_word = [(i, word[0]) for i, word in
                       enumerate(sorted(w_dict.items(), key=lambda x: x[1], reverse=True))]
        vocab = {w: i for i, w in int_to_word}
        return [[vocab[w.lower()] if lowercase else vocab[w]
                 for w in sen] for sen in texts], vocab
