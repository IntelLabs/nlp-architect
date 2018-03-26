from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
import os

import numpy as np
from keras.preprocessing.sequence import pad_sequences

from utils import (Vocabulary, add_offset, download_file, fill_embedding_mat,
                   load_word_embeddings, one_hot, one_hot_sentence, unzip_file)


class IntentDataset(object):
    """
    Intent extraction base class

    Arguments:
        url(str): URL of dataset
        filename(str): dataset file to download from URL
        path(str): local path for saving dataset
        sentence_length(int): MAX sentence length
    """
    def __init__(self, url, filename, path, sentence_length=30,  embedding_model=None,
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

    def load_embedding(self, files):
        print('Loading external word embedding model ..')
        emb_vecs = load_word_embeddings(self.embedding_model, self.embedding_size)
        for f in files:
            self.vecs[f][0] = fill_embedding_mat(self.vecs[f][0],
                                                 self._tokens_vocab.reverse_vocab(),
                                                 emb_vecs,
                                                 self.embedding_size)

    def load_data(self, files, file_format):
        # read files and parse
        for f in files:
            with open(self.path + os.sep + file_format.format(f)) as fp:
                data = fp.readlines()
                sentences = self.split_into_sentences(data)
                self.data_dict[f] = self.parse_sentences(sentences)

        # vectorize
        # add offset of 2 for PAD and OOV
        self._tokens_vocab.add_vocab_offset(2)
        self._slots_vocab.add_vocab_offset(1)
        for f in files:
            x = pad_sequences(add_offset(
                self.data_dict[f][0], 2), maxlen=self.sentence_len)
            _y = pad_sequences(add_offset(
                self.data_dict[f][2]), maxlen=self.sentence_len)
            y = one_hot_sentence(_y, self.slot_vocab_size)
            i = one_hot(self.data_dict[f][1], self.intent_size)
            self.vecs[f] = [x, i, y]

    @staticmethod
    def split_into_sentences(file_lines):
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

    def parse_sentences(self, sentences):
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
        return len(self._tokens_vocab) + 2

    @property
    def slot_vocab_size(self):
        return len(self._slots_vocab) + 1

    @property
    def intent_size(self):
        return len(self._intents_vocab)

    @property
    def tokens_vocab(self):
        return self._tokens_vocab.vocab

    @property
    def slots_vocab(self):
        return self._slots_vocab.vocab

    @property
    def intents_vocab(self):
        return self._intents_vocab.vocab

    @property
    def train_set(self):
        return self.vecs['train']

    @property
    def test_set(self):
        return self.vecs['test']


class ATIS(IntentDataset):
    """
    ATIS dataset (numpy arrays format)
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

        self.load_data(self.files, self.file_format)
        if self.embedding_model is not None:
            self.load_embedding(self.files)


class SNIPS(IntentDataset):
    """
    SNIPS dataset class
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

        self.load_data(self.files, self.file_format)
        if self.embedding_model is not None:
            self.load_embedding(self.files)
