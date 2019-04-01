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

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
import sys

import numpy as np
from nlp_architect.utils.generic import pad_sentences
from nlp_architect.utils.text import SpacyInstance, Vocabulary, character_vector_generator, \
    word_vector_generator


class IntentDataset(object):
    """
    Intent extraction dataset base class

    Args:
        sentence_length (int): max sentence length
    """

    def __init__(self, sentence_length=50, word_length=12):
        self.data_dict = {}
        self.vecs = {}
        self.sentence_len = sentence_length
        self.word_len = word_length

        self._tokens_vocab = Vocabulary(2)
        self._chars_vocab = Vocabulary(2)
        self._tags_vocab = Vocabulary(1)
        self._intents_vocab = Vocabulary()

    def _load_data(self, train_set, test_set):
        # vectorize
        # add offset of 2 for PAD and OOV
        train_size = len(train_set)
        test_size = len(test_set)
        texts, tags, intents = list(zip(*train_set + test_set))
        text_vectors, self._tokens_vocab = word_vector_generator(texts, lower=True, start=2)
        tag_vectors, self._tags_vocab = word_vector_generator(tags, lower=False, start=1)
        chars_vectors, self._chars_vocab = character_vector_generator(texts, start=2)
        i, self._intents_vocab = word_vector_generator([intents])
        i = np.asarray(i[0])

        text_vectors = pad_sentences(text_vectors, max_length=self.sentence_len)
        tag_vectors = pad_sentences(tag_vectors, max_length=self.sentence_len)
        chars_vectors = [pad_sentences(d, max_length=self.word_len) for d in chars_vectors]
        zeros = np.zeros((len(chars_vectors), self.sentence_len, self.word_len))
        for idx, d in enumerate(chars_vectors):
            d = d[:self.sentence_len]
            zeros[idx, :d.shape[0]] = d
        chars_vectors = zeros.astype(dtype=np.int32)

        self.vecs['train'] = [text_vectors[:train_size],
                              chars_vectors[:train_size],
                              i[:train_size],
                              tag_vectors[:train_size]]
        self.vecs['test'] = [text_vectors[-test_size:],
                             chars_vectors[-test_size:],
                             i[-test_size:],
                             tag_vectors[-test_size:]]

    @property
    def word_vocab_size(self):
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
    def word_vocab(self):
        """dict: tokens vocabulary"""
        return self._tokens_vocab

    @property
    def char_vocab(self):
        """dict: word character vocabulary"""
        return self._chars_vocab

    @property
    def tags_vocab(self):
        """dict: labels vocabulary"""
        return self._tags_vocab

    @property
    def intents_vocab(self):
        """dict: intent labels vocabulary"""
        return self._intents_vocab

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
    - one word per line with tag annotation and intent type separated
    by tabs <token>\t<tag_label>\t<intent>\n
    - sentences are separated by an empty line

    Args:
        train_file (str): path to train set file
        test_file (str): path to test set file
        sentence_length (int): max sentence length
        word_length (int): max word length
    """
    files = ['train', 'test']

    def __init__(self, train_file, test_file, sentence_length=30, word_length=12):
        train_set_raw, test_set_raw = self._load_dataset(train_file, test_file)
        super(TabularIntentDataset, self).__init__(sentence_length=sentence_length,
                                                   word_length=word_length)

        self._load_data(train_set_raw, test_set_raw)

    def _load_dataset(self, train_file, test_file):
        """returns a tuple of train/test with 3-tuple of tokens, tags, intent_type"""
        train = self._parse_sentences(self._read_file(train_file))
        test = self._parse_sentences(self._read_file(test_file))
        return train, test

    def _read_file(self, path):
        with open(path, encoding='utf-8', errors='ignore') as fp:
            data = fp.readlines()
        return self._split_into_sentences(data)

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
        if s:
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
            path (str): dataset path
            sentence_length (int, optional): max sentence length
            word_length (int, optional): max word length
    """
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

    def __init__(self, path, sentence_length=30, word_length=12):
        if path is None or not os.path.isdir(path):
            print('invalid path for SNIPS dataset loader')
            sys.exit(0)
        self.dataset_root = path
        train_set_raw, test_set_raw = self._load_dataset()
        super(SNIPS, self).__init__(sentence_length=sentence_length,
                                    word_length=word_length)
        self._load_data(train_set_raw, test_set_raw)

    def _load_dataset(self):
        """returns a tuple of train/test with 3-tuple of tokens, tags, intent_type"""
        train_data = self._load_intents(self.train_files)
        test_data = self._load_intents(self.test_files)
        train = [(t, l, i) for i in sorted(train_data) for t, l in train_data[i]]
        test = [(t, l, i) for i in sorted(test_data) for t, l in test_data[i]]
        return train, test

    def _load_intents(self, files):
        data = {}
        for f in sorted(files):
            fname = os.path.join(self.dataset_root, f)
            intent = f.split(os.sep)[0]
            with open(fname, encoding='utf-8', errors='ignore') as fp:
                fdata = json.load(fp)
            entries = self._parse_json([d['data'] for d in fdata[intent]])
            data[intent] = entries
        return data

    def _parse_json(self, data):
        tok = SpacyInstance(disable=['tagger', 'ner', 'parser', 'vectors', 'textcat'])
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
