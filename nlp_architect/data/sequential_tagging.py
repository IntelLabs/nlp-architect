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

from os import path

import numpy as np

from nlp_architect.utils.generic import pad_sentences
from nlp_architect.utils.io import validate_existing_directory, validate_existing_filepath
from nlp_architect.utils.text import read_sequential_tagging_file, \
    word_vector_generator, character_vector_generator


class SequentialTaggingDataset(object):
    """
    Sequential tagging dataset loader.
    Loads train/test files with tabular separation.

    Args:
        train_file (str): path to train file
        test_file (str): path to test file
        max_sentence_length (int, optional): max sentence length
        max_word_length (int, optional): max word length
        tag_field_no (int, optional): index of column to use a y-samples
    """

    def __init__(self,
                 train_file,
                 test_file,
                 max_sentence_length=30,
                 max_word_length=20,
                 tag_field_no=2):
        self.files = {'train': train_file,
                      'test': test_file}
        self.max_sent_len = max_sentence_length
        self.max_word_len = max_word_length
        self.tf = tag_field_no

        self.vocabs = {'token': None,
                       'char': None,   # 0=pad, 1=unk
                       'tag': None}    # 0=pad

        self.data = {}

        sentences = self._read_file(self.files['train'])
        train_size = len(sentences)
        sentences += self._read_file(self.files['test'])
        test_size = len(sentences) - train_size
        texts, tags = list(zip(*sentences))

        texts_mat, self.vocabs['token'] = word_vector_generator(texts, lower=True, start=2)
        tags_mat, self.vocabs['tag'] = word_vector_generator(tags, start=1)
        chars_mat, self.vocabs['char'] = character_vector_generator(texts, start=2)

        texts_mat = pad_sentences(texts_mat, max_length=self.max_sent_len)
        tags_mat = pad_sentences(tags_mat, max_length=self.max_sent_len)

        chars_mat = [pad_sentences(d, max_length=self.max_word_len) for d in chars_mat]
        zeros = np.zeros((len(chars_mat), self.max_sent_len, self.max_word_len))
        for idx, d in enumerate(chars_mat):
            d = d[:self.max_sent_len]
            zeros[idx, :d.shape[0]] = d
        chars_mat = zeros.astype(dtype=np.int32)

        self.data['train'] = texts_mat[:train_size], chars_mat[:train_size], tags_mat[:train_size]
        self.data['test'] = texts_mat[-test_size:], chars_mat[-test_size:], tags_mat[-test_size:]

    @property
    def y_labels(self):
        """return y labels"""
        return self.vocabs['tag']

    @property
    def word_vocab(self):
        """words vocabulary"""
        return self.vocabs['token']

    @property
    def char_vocab(self):
        """characters vocabulary"""
        return self.vocabs['char']

    @property
    def word_vocab_size(self):
        """word vocabulary size"""
        return len(self.vocabs['token']) + 2

    @property
    def char_vocab_size(self):
        """character vocabulary size"""
        return len(self.vocabs['char']) + 2

    @property
    def train_set(self):
        """Get the train set"""
        return self.data['train']

    @property
    def test_set(self):
        """Get the test set"""
        return self.data['test']

    def _read_file(self, filepath):
        with open(filepath, encoding='utf-8') as fp:
            data = fp.readlines()
            data = [d.strip() for d in data]
            sentences = self._split_into_sentences(data)
            parsed_sentences = [self._parse_sentence(s) for s in sentences if len(s) > 0]
        return parsed_sentences

    def _parse_sentence(self, sentence):
        tokens = []
        tags = []
        for line in sentence:
            fields = line.split()
            assert len(fields) >= self.tf, 'tag field exceeds number of fields'
            if 'CD' in fields[1]:
                tokens.append('0')
            else:
                tokens.append(fields[0])
            tags.append(fields[self.tf - 1])
        return tokens, tags

    @staticmethod
    def _split_into_sentences(file_lines):
        sents = []
        s = []
        for line in file_lines:
            line = line.strip()
            if not line:
                sents.append(s)
                s = []
                continue
            s.append(line)
        sents.append(s)
        return sents


class CONLL2000(object):
    """
        CONLL 2000 POS/chunking task data set (numpy)

        Arguments:
            data_path (str): directory containing CONLL2000 files
            sentence_length (int, optional): number of time steps to embed the data.
                None value will not truncate vectors
            max_word_length (int, optional): max word length in characters.
                None value will not truncate vectors
            extract_chars (boolean, optional): Yield Char RNN features.
            lowercase (bool, optional): lower case sentence words
        """

    dataset_files = {'train': 'train.txt',
                     'test': 'test.txt'}

    def __init__(self,
                 data_path,
                 sentence_length=None,
                 max_word_length=None,
                 extract_chars=False,
                 lowercase=True):
        self._validate_paths(data_path)
        self.data_path = data_path
        self.sentence_length = sentence_length
        self.use_chars = extract_chars
        self.max_word_length = max_word_length
        self.lower = lowercase
        self.vocabs = {'word': None,
                       'char': None,
                       'pos': None,
                       'chunk': None}
        self._data_dict = {}

    def _validate_paths(self, data_path):
        validate_existing_directory(data_path)
        for f in self.dataset_files:
            _f_path = path.join(data_path, self.dataset_files[f])
            validate_existing_filepath(_f_path)
            self.dataset_files[f] = _f_path

    def _load_data(self):
        """
        open files and parse
        return format: list of 3-tuples (word list, POS list, chunk list)
        """
        train_set = read_sequential_tagging_file(self.dataset_files['train'])
        test_set = read_sequential_tagging_file(self.dataset_files['test'])
        train_data = [list(zip(*x)) for x in train_set]
        test_data = [list(zip(*x)) for x in test_set]
        return train_data, test_data

    @property
    def train_set(self):
        """get the train set"""
        if self._data_dict.get('train', None) is None:
            self._gen_data()
        return self._data_dict.get('train')

    @property
    def test_set(self):
        """get the test set"""
        if self._data_dict.get('test', None) is None:
            self._gen_data()
        return self._data_dict.get('test')

    @staticmethod
    def _extract(x, y, n):
        return list(zip(*x))[n] + list(zip(*y))[n]

    @property
    def word_vocab(self):
        """word Vocabulary"""
        return self.vocabs['word']

    @property
    def char_vocab(self):
        """character Vocabulary"""
        return self.vocabs['char']

    @property
    def pos_vocab(self):
        """pos label Vocabulary"""
        return self.vocabs['pos']

    @property
    def chunk_vocab(self):
        """chunk label Vocabulary"""
        return self.vocabs['chunk']

    def _gen_data(self):
        train, test = self._load_data()
        train_size = len(train)
        test_size = len(test)
        sentences = self._extract(train, test, 0)
        pos_tags = self._extract(train, test, 1)
        chunk_tags = self._extract(train, test, 2)
        sentence_vecs, word_vocab = word_vector_generator(sentences, self.lower, 2)
        pos_vecs, pos_vocab = word_vector_generator(pos_tags, start=1)
        chunk_vecs, chunk_vocab = word_vector_generator(chunk_tags, start=1)
        self.vocabs = {'word': word_vocab,  # 0=pad, 1=unk
                       'pos': pos_vocab,  # 0=pad, 1=unk
                       'chunk': chunk_vocab}  # 0=pad
        if self.sentence_length is not None:
            sentence_vecs = pad_sentences(sentence_vecs, max_length=self.sentence_length)
            chunk_vecs = pad_sentences(chunk_vecs, max_length=self.sentence_length)
            pos_vecs = pad_sentences(pos_vecs, max_length=self.sentence_length)
        self._data_dict['train'] = sentence_vecs[:train_size], pos_vecs[:train_size], \
            chunk_vecs[:train_size]
        self._data_dict['test'] = sentence_vecs[-test_size:], pos_vecs[-test_size:], \
            chunk_vecs[-test_size:]
        if self.use_chars:
            chars_vecs, char_vocab = character_vector_generator(sentences, start=2)
            self.vocabs.update({'char': char_vocab})  # 0=pad, 1=unk
            if self.max_word_length is not None:
                chars_vecs = [pad_sentences(d, max_length=self.max_word_length)
                              for d in chars_vecs]
                zeros = np.zeros((len(chars_vecs), self.sentence_length, self.max_word_length))
                for idx, d in enumerate(chars_vecs):
                    d = d[:self.sentence_length]
                    zeros[idx, -d.shape[0]:] = d
                chars_vecs = zeros.astype(dtype=np.int32)
            self._data_dict['train'] += (chars_vecs[:train_size],)
            self._data_dict['test'] += (chars_vecs[-test_size:],)
