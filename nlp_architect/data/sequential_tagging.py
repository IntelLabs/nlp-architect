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

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
from keras.preprocessing.sequence import pad_sequences

from nlp_architect.utils.text import Vocabulary


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
                 tag_field_no=4):
        self.files = {'train': train_file,
                      'test': test_file}
        self.max_sent_len = max_sentence_length
        self.max_word_len = max_word_length
        self.tf = tag_field_no

        self.vocabs = {'token': Vocabulary(2),  # 0=pad, 1=unk
                       'char': Vocabulary(2),   # 0=pad, 1=unk
                       'tag': Vocabulary(1)}    # 0=pad

        self.data = {}
        for f in self.files:
            raw_sentences = self._read_file(self.files[f])
            word_vecs = []
            char_vecs = []
            tag_vecs = []
            for tokens, tags in raw_sentences:
                word_vecs.append(np.array([self.vocabs['token'].add(t) for t in tokens]))
                word_chars = []
                for t in tokens:
                    word_chars.append(np.array([self.vocabs['char'].add(c) for c in t]))
                word_chars = pad_sequences(word_chars, maxlen=self.max_word_len)
                if self.max_sent_len - len(tokens) > 0:
                    char_padding = self.max_sent_len - len(word_chars)
                    char_vecs.append(
                        np.concatenate((np.zeros((char_padding, self.max_word_len)), word_chars),
                                       axis=0))
                else:
                    char_vecs.append(word_chars[-self.max_sent_len:])
                tag_vecs.append(np.array([self.vocabs['tag'].add(t) for t in tags]))
            word_vecs = pad_sequences(word_vecs, maxlen=self.max_sent_len)
            char_vecs = np.asarray(char_vecs)
            tag_vecs = pad_sequences(tag_vecs, maxlen=self.max_sent_len)
            self.data[f] = word_vecs, char_vecs, tag_vecs

    @property
    def y_labels(self):
        """return y labels"""
        return self.vocabs['tag'].vocab

    @property
    def word_vocab(self):
        """words vocabulary"""
        return self.vocabs['token'].vocab

    @property
    def char_vocab(self):
        """characters vocabulary"""
        return self.vocabs['char'].vocab

    @property
    def word_vocab_size(self):
        """word vocabulary size"""
        return len(self.vocabs['token']) + 2

    @property
    def char_vocab_size(self):
        """character vocabulary size"""
        return len(self.vocabs['char']) + 2

    @property
    def train(self):
        """Get the train set"""
        return self.data['train']

    @property
    def test(self):
        """Get the test set"""
        return self.data['test']

    def _read_file(self, path):
        with open(path, encoding='utf-8') as fp:
            data = fp.readlines()
            data = [d.strip() for d in data]
            data = [d for d in data if 'DOCSTART' not in d]
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
