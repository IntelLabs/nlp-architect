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

import json
import os
import sys

import numpy as np
from keras.preprocessing.sequence import pad_sequences

from nlp_architect.utils.embedding import load_word_embeddings, fill_embedding_mat
from nlp_architect.utils.generic import one_hot_sentence, one_hot
from nlp_architect.utils.io import download_file
from nlp_architect.utils.text import Vocabulary, SpacyTokenizer


class IntentDataset(object):
    """
    Intent extraction dataset base class

    Args:
        sentence_length (int): max sentence length
        embedding_model (str, optional): external embedding model path
        embedding_size (int): embedding vectors size
    """

    def __init__(self, sentence_length, word_length=12, embedding_model=None,
                 embedding_size=None):
        self.data_dict = {}
        self.vecs = {}
        self.sentence_len = sentence_length
        self.word_len = word_length
        self.embedding_model = embedding_model
        self.embedding_size = embedding_size

        self._tokens_vocab = Vocabulary()
        self._chars_vocab = Vocabulary()
        self._tags_vocab = Vocabulary()
        self._intents_vocab = Vocabulary()

    def _load_embedding(self, files):
        print('Loading external word embedding model ..')
        emb_vecs, _ = load_word_embeddings(self.embedding_model)
        for f in files:
            self.vecs[f][0] = fill_embedding_mat(self.vecs[f][0],
                                                 self._tokens_vocab.reverse_vocab(),
                                                 emb_vecs,
                                                 self.embedding_size)

    def _load_data(self, train_set, test_set):
        datasets = {'train': train_set, 'test': test_set}
        # vectorize
        # add offset of 2 for PAD and OOV
        self._tokens_vocab.add_vocab_offset(2)
        self._chars_vocab.add_vocab_offset(2)
        self._tags_vocab.add_vocab_offset(1)
        vec_data = {}
        for f in datasets.keys():
            vec_data[f] = self._prepare_vectors(datasets[f])
        for f in datasets.keys():
            tokens, words, intents, tags = vec_data[f]
            x = pad_sequences(tokens, maxlen=self.sentence_len)
            _w = []
            for s in words:
                _s = pad_sequences(s, maxlen=self.word_len)
                sentence = np.asarray(_s)[-self.sentence_len:]
                if sentence.shape[0] < self.sentence_len:
                    sentence = np.vstack((np.zeros((self.sentence_len - sentence.shape[0],
                                                    self.word_len)), sentence))
                _w.append(sentence)
            w = np.asarray(_w)
            _y = pad_sequences(tags, maxlen=self.sentence_len)
            y = one_hot_sentence(_y, self.label_vocab_size)
            i = one_hot(intents, self.intent_size)
            self.vecs[f] = [x, w, i, y]

    def _prepare_vectors(self, dataset):
        tokens = []
        words = []
        tags = []
        intents = []
        for tok, tag, i in dataset:
            tokens.append(np.asarray([self._tokens_vocab.add(t) for t in tok]))
            words.append(np.asarray(self._extract_char_features(tok)))
            tags.append(np.asarray([self._tags_vocab.add(t) for t in tag]))
            intents.append(self._intents_vocab.add(i))
        return tokens, words, np.asarray(intents), tags

    def _extract_char_features(self, tokens):
        words = []
        for t in tokens:
            words.append(np.asarray([self._chars_vocab.add(c) for c in t]))
        return words

    @property
    def vocab_size(self):
        """int: vocabulary size"""
        return len(self._tokens_vocab) + 2

    @property
    def char_vocab_size(self):
        """int: char vocabulary size"""
        return len(self._chars_vocab) + 2

    @property
    def label_vocab_size(self):
        """int: label vocabulary size"""
        return len(self._tags_vocab) + 1

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
        return self._tags_vocab.vocab

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


class TabularIntentDataset(IntentDataset):
    """
    Tabular Intent/Slot tags dataset loader.
    Compatible with many sequence tagging datasets (ATIS, CoNLL, etc..)
    data format must be int tabular format where:
        - one word per line with tag annotation and intent type separated by tabs
            <token>\t<tag_label>\t<intent>\n
        - sentences are separated by an empty line

    Args:
            train_file (str): path to train set file
            test_file (str): path to test set file
            sentence_length (int): max sentence length
            word_length (int): max word length
            embedding_model (str): external word embedding model path
            embedding_size (int): external word embedding vector size
    """
    files = ['train', 'test']

    def __init__(self, train_file, test_file, sentence_length=30, word_length=12,
                 embedding_model=None, embedding_size=None):
        train_set_raw, test_set_raw = self._load_dataset(train_file, test_file)
        super(TabularIntentDataset, self).__init__(sentence_length=sentence_length,
                                                   word_length=word_length,
                                                   embedding_model=embedding_model,
                                                   embedding_size=embedding_size)

        self._load_data(train_set_raw, test_set_raw)
        if self.embedding_model is not None:
            self._load_embedding(self.files)

    def _load_dataset(self, train_file, test_file):
        """returns a tuple of train/test with 3-tuple of tokens, tags, intent_type"""
        train = self._parse_sentences(self._read_file(train_file))
        test = self._parse_sentences(self._read_file(test_file))
        return train, test

    def _read_file(self, path):
        with open(path, errors='ignore') as fp:
            data = fp.readlines()
        return self._split_into_sentences(data)

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

    @staticmethod
    def _parse_sentences(sentences):
        encoded_sentences = []
        for sen in sentences:
            tokens = []
            tags = []
            intent = None
            for line in sen:
                t, s, i = line.split()
                tokens.append(t)
                tags.append(s)
                intent = i
                if intent is None:
                    intent = i
            encoded_sentences.append((tokens, tags, intent))
        return encoded_sentences


class SNIPS(IntentDataset):
    """
    SNIPS dataset class

    Args:
            sentence_length (int, optional): max sentence length
            word_length (int, optional): max word length
            path (str, optional): path where to save dataset files
            embedding_model (str, optional): external word embedding model path
            embedding_size (int, optional): external word embedding vector size
    """
    url = 'https://raw.githubusercontent.com/snipsco/nlu-benchmark/master/2017-06' \
          '-custom-intent-engines/'
    train_files = [
        'AddToPlaylist/train_AddToPlaylist_full.json',
        'BookRestaurant/train_BookRestaurant_full.json',
        'GetWeather/train_GetWeather_full.json',
        'PlayMusic/train_PlayMusic_full.json',
        'RateBook/train_RateBook_full.json',
        'SearchCreativeWork/train_SearchCreativeWork_full.json',
        'SearchScreeningEvent/train_SearchScreeningEvent_full.json'
    ]
    test_files = [
        'AddToPlaylist/validate_AddToPlaylist.json',
        'BookRestaurant/validate_BookRestaurant.json',
        'GetWeather/validate_GetWeather.json',
        'PlayMusic/validate_PlayMusic.json',
        'RateBook/validate_RateBook.json',
        'SearchCreativeWork/validate_SearchCreativeWork.json',
        'SearchScreeningEvent/validate_SearchScreeningEvent.json'
    ]
    files = ['train', 'test']

    def __init__(self, sentence_length=30, word_length=12, path=None, embedding_model=None,
                 embedding_size=None):
        if path is None:
            self.dataset_root = os.path.expanduser('~') + '/nlp_architect/data/snips/'
        self._download_dataset()
        train_set_raw, test_set_raw = self._load_dataset()
        super(SNIPS, self).__init__(sentence_length=sentence_length,
                                    word_length=word_length,
                                    embedding_model=embedding_model,
                                    embedding_size=embedding_size)
        self._load_data(train_set_raw, test_set_raw)
        if self.embedding_model is not None:
            self._load_embedding(self.files)

    def _download_dataset(self):
        if not os.path.exists(self.dataset_root):
            self._ask_if_download()
            os.makedirs(self.dataset_root)
        for f in self.train_files + self.test_files:
            fp = self.dataset_root + f
            if not os.path.exists(os.path.dirname(fp)):
                os.makedirs(os.path.dirname(fp))
            if not os.path.exists(fp):
                download_file(self.url, f, fp)

    def _load_dataset(self):
        """returns a tuple of train/test with 3-tuple of tokens, tags, intent_type"""
        train_data = self._load_intents(self.train_files)
        test_data = self._load_intents(self.test_files)
        train = [(t, l, i) for i in train_data for t, l in train_data[i]]
        test = [(t, l, i) for i in test_data for t, l in test_data[i]]
        return train, test

    def _load_intents(self, files):
        data = {}
        for f in files:
            fname = self.dataset_root + f
            intent = f.split(os.sep)[0]
            fdata = json.load(open(fname, encoding='utf-8', errors='ignore'))
            entries = self._parse_json([d['data'] for d in fdata[intent]])
            data[intent] = entries
        return data

    def _parse_json(self, data):
        tok = SpacyTokenizer()
        sentences = []
        for s in data:
            tokens = []
            tags = []
            for t in s:
                new_tokens = tok.tokenize(t['text'].strip())
                tokens += new_tokens
                ent = t.get('entity', None)
                if ent is not None:
                    tags += self._create_tags(ent, len(new_tokens))
                else:
                    tags += ['O'] * len(new_tokens)
            sentences.append((tokens, tags))
        return sentences

    @staticmethod
    def _create_tags(tag, length):
        labels = ['B-' + tag]
        if length > 1:
            for _ in range(length - 1):
                labels.append('I-' + tag)
        return labels

    @staticmethod
    def _ask_if_download():
        url = 'https://github.com/snipsco/nlu-benchmark'
        print('SNIPS NLU benchmark dataset was not found')
        print('License: CC0-1.0 https://github.com/snipsco/nlu-benchmark/blob/master/LICENSE')
        response = input('To download the dataset from {}, please type YES: '.format(url))
        if response.lower().strip() == "yes":
            print('The terms and conditions of the data set license apply. Intel does not '
                  + 'grant any rights to the data files or database')
            print('Proceeding to download the dataset')
            return
        else:
            print('Download declined. Response received {} != YES.'.format(response))
            print('Download data files manually and provide path in \'path\' keyword of '
                  'this class')
            sys.exit(0)
