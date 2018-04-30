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
import os

import numpy as np
from keras.preprocessing.sequence import pad_sequences

from nlp_architect.utils.embedding import load_word_embeddings, fill_embedding_mat
from nlp_architect.utils.io import download_file, unzip_file
from nlp_architect.utils.generic import add_offset, one_hot_sentence, one_hot
from nlp_architect.utils.text import Vocabulary


class IntentDataset(object):
    """
    Intent extraction dataset base class

    Args:
        url (str): URL of dataset
        filename (str): dataset file to download from URL
        path (str): local path for saving dataset
        sentence_length (int, optional): max sentence length
        embedding_model (str, optional): external embedding model path
        embedding_size (int): embedding vectors size
    """
    def __init__(self, url, filename, path, sentence_length=30, embedding_model=None,
                 embedding_size=None):
        self.data_dict = {}
        self.vecs = {}
        self.url = url
        self.file = filename
        self.datafile = path + os.sep + self.file
        self.path = path
        self.sentence_len = sentence_length
        self.embedding_model = embedding_model
        self.embedding_size = embedding_size

        self._tokens_vocab = Vocabulary()
        self._slots_vocab = Vocabulary()
        self._intents_vocab = Vocabulary()

        if not os.path.exists(self.datafile):
            download_file(self.url, self.file, self.datafile)
            unzip_file(self.datafile)

    def _load_embedding(self, files):
        print('Loading external word embedding model ..')
        emb_vecs = load_word_embeddings(self.embedding_model, self.embedding_size)
        for f in files:
            self.vecs[f][0] = fill_embedding_mat(self.vecs[f][0],
                                                 self._tokens_vocab.reverse_vocab(),
                                                 emb_vecs,
                                                 self.embedding_size)

    def _load_data(self, files, file_format):
        # read files and parse
        for f in files:
            with open(self.path + os.sep + file_format.format(f)) as fp:
                data = fp.readlines()
                sentences = self._split_into_sentences(data)
                self.data_dict[f] = self._parse_sentences(sentences)

        # vectorize
        # add offset of 2 for PAD and OOV
        self._tokens_vocab.add_vocab_offset(2)
        self._slots_vocab.add_vocab_offset(1)
        for f in files:
            x = pad_sequences(add_offset(
                self.data_dict[f][0], 2), maxlen=self.sentence_len)
            _y = pad_sequences(add_offset(
                self.data_dict[f][2]), maxlen=self.sentence_len)
            y = one_hot_sentence(_y, self.label_vocab_size)
            i = one_hot(self.data_dict[f][1], self.intent_size)
            self.vecs[f] = [x, i, y]

    @staticmethod
    def _split_into_sentences(file_lines):
        sents = []
        s = []
        for line in file_lines:
            line = line.strip()
            if len(line) == 0:
                sents.append(s)
                s = []
                continue
            s.append(line)
        else:
            sents.append(s)
        return sents

    def _parse_sentences(self, sentences):
        tokens = []
        tags = []
        intents = []
        for sen in sentences:
            token = []
            tag = []
            intent = None
            for line in sen:
                t, s, i = line.split('\t')
                token.append(self._tokens_vocab.add(t))
                tag.append(self._slots_vocab.add(s))
                if intent is None:
                    intent = self._intents_vocab.add(i)
            tokens.append(np.array(token))
            tags.append(np.array(tag))
            intents.append(intent)
        return np.array(tokens), np.array(intents), np.array(tags)

    @property
    def vocab_size(self):
        """int: vocabulary size"""
        return len(self._tokens_vocab) + 2

    @property
    def label_vocab_size(self):
        """int: label vocabulary size"""
        return len(self._slots_vocab) + 1

    @property
    def intent_size(self):
        """int: intent label vocabulary size"""
        return len(self._intents_vocab)

    @property
    def tokens_vocab(self):
        """dict: tokens vocabulary"""
        return self._tokens_vocab.vocab

    @property
    def labels_vocab(self):
        """dict: labels vocabulary"""
        return self._slots_vocab.vocab

    @property
    def intents_vocab(self):
        """dict: intent labels vocabulary"""
        return self._intents_vocab.vocab

    @property
    def train_set(self):
        """:obj:`tuple` of :obj:`numpy.ndarray`: train set"""
        return self.vecs['train']

    @property
    def test_set(self):
        """:obj:`tuple` of :obj:`numpy.ndarray`: test set"""
        return self.vecs['test']


class ATIS(IntentDataset):
    """
    ATIS dataset

    Args:
            sentence_length (int): max sentence length
            path (str): path where to save dataset files
            embedding_model (str): external word embedding model path
            embedding_size (int): external word embedding vector size
    """
    files = ['train',
             'test']
    file_format = 'atis/atis-{}.txt'

    def __init__(self, sentence_length=30, path='.', embedding_model=None, embedding_size=None):
        self.url = 'https://s3-us-west-1.amazonaws.com/nervana-modelzoo/intent_extraction'
        self.file = 'atis.zip'
        super(ATIS, self).__init__(url=self.url, filename=self.file, path=path,
                                   sentence_length=sentence_length,
                                   embedding_model=embedding_model,
                                   embedding_size=embedding_size)

        self._load_data(self.files, self.file_format)
        if self.embedding_model is not None:
            self._load_embedding(self.files)


class SNIPS(IntentDataset):
    """
    SNIPS dataset class

    Args:
            sentence_length (int): max sentence length
            path (str): path where to save dataset files
            embedding_model (str): external word embedding model path
            embedding_size (int): external word embedding vector size
    """
    files = ['train',
             'test']
    file_format = 'snips/snips-{}.txt'

    def __init__(self, sentence_length=30, path='.', embedding_model=None, embedding_size=None):
        self.url = 'https://s3-us-west-1.amazonaws.com/nervana-modelzoo/intent_extraction'
        self.file = 'snips.zip'
        super(SNIPS, self).__init__(url=self.url, filename=self.file, path=path,
                                    sentence_length=sentence_length,
                                    embedding_model=embedding_model,
                                    embedding_size=embedding_size)

        self._load_data(self.files, self.file_format)
        if self.embedding_model is not None:
            self._load_embedding(self.files)
