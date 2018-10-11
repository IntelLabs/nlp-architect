#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2018/9/20 23:34

__author__ = 'xujiang@baixing.com'

import os
import pickle
from itertools import chain
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .sentence import Sentence
from .sentence import TagPrefix
from .sentence import TagSurfix


class DataHandler(object):
    def __init__(self, rootDir='There is no corpus finded...', save_path=' '):
        self.rootDir = rootDir
        self.save_path = save_path
        self.spiltChar = ['。', '!', '！', '？', '?']
        self.max_len = 200
        self.totalLine = 0
        self.longLine = 0
        self.totalChars = 0
        self.TAGPRE = TagPrefix.convert()

    def loadData(self):
        isFile = os.path.isfile(self.save_path)

        if isFile:
            with open(self.save_path, 'rb') as inp:
                self.X = pickle.load(inp)
                self.y = pickle.load(inp)
                self.word2id = pickle.load(inp)
                if self.save_path == 'data/data_ner_0409.pkl':
                    self.id2word = pickle.load(inp)
                self.tag2id = pickle.load(inp)
                if self.save_path == 'data/data_ner_0409.pkl':
                    self.id2tag = pickle.load(inp)
        else:
            self.loadRawData()
            self.handlerRawData()

    def loadRawData(self):
        self.datas = list()
        self.labels = list()
        if self.rootDir:
            print(self.rootDir)
            for dirName, subdirList, fileList in os.walk(self.rootDir):
                # curDir = os.path.join(self.rootDir, dirName)
                for file in fileList:
                    if file.endswith(".txt"):
                        curFile = os.path.join(dirName, file)
                        print("processing:%s" % (curFile))
                        with open(curFile, "r", encoding='utf-8') as fp:
                            for line in fp.readlines():
                                self.processLine(line)

            print("total:%d, long lines:%d, total chars:%d" % (self.totalLine, self.longLine, self.totalChars))
            print('Length of datas is %d' % len(self.datas))
            print('Example of datas: ', self.datas[0])
            print('Example of labels:', self.labels[0])

    def processLine(self, line):
        line = line.strip()
        nn = len(line)
        seeLeftB = False
        start = 0
        sentence = Sentence()
        try:
            for i in range(nn):
                if line[i] == ' ':
                    if not seeLeftB:
                        token = line[start:i]
                        if token.startswith('['):
                            token_ = ''
                            for j in [i.split('/') for i in token.split('[')[1].split(']')[0].split(' ')]:
                                token_ += j[0]
                            token_ = token_ + '/' + token.split('/')[-1]
                            self.processToken(token_, sentence, False)
                        else:
                            self.processToken(token, sentence, False)
                        start = i + 1
                elif line[i] == '[':
                    seeLeftB = True
                elif line[i] == ']':
                    seeLeftB = False
            # 此部分未与上面处理方式统一，（小概率事件）数据多元化，增加模型泛化能力。
            if start < nn:
                token = line[start:]
                if token.startswith('['):
                    tokenLen = len(token)
                    while tokenLen > 0 and token[tokenLen - 1] != ']':
                        tokenLen = tokenLen - 1
                    token = token[1:tokenLen - 1]
                    ss = token.split(' ')
                    ns = len(ss)
                    for i in range(ns - 1):
                        self.processToken(ss[i], sentence, False)
                    self.processToken(ss[-1], sentence, True)
                else:
                    self.processToken(token, sentence, True)
        except Exception as e:
            print('处理数据异常, 异常行为：' + line)
            print(e)

    def processToken(self, tokens, sentence, end):
        nn = len(tokens)
        while nn > 0 and tokens[nn - 1] != '/':
            nn = nn - 1

        token = tokens[:nn - 1].strip()
        tagPre = tokens[nn:].strip()
        tagPre = self.TAGPRE.get(tagPre, TagPrefix.general.value)
        if token not in self.spiltChar:
            sentence.addToken(token, tagPre)
        if token in self.spiltChar or end:
            if sentence.chars > self.max_len:
                self.longLine += 1
            else:
                x = []
                y = []
                self.totalChars += sentence.chars
                sentence.generate_tr_line(x, y)

                if len(x) > 0 and len(x) == len(y):
                    self.datas.append(x)
                    self.labels.append(y)
                else:
                    print('处理一行数据异常, 异常行如下')
                    print(sentence.tokens)
            self.totalLine += 1
            sentence.clear()

    def handlerRawData(self):
        self.df_data = pd.DataFrame({'words': self.datas, 'tags': self.labels}, index=range(len(self.datas)))
        self.df_data['sentence_len'] = self.df_data['words'].apply(
            lambda words: len(words))

        all_words = list(chain(*self.df_data['words'].values))
        sr_allwords = pd.Series(all_words)
        sr_allwords = sr_allwords.value_counts()

        set_words = sr_allwords.index
        set_ids = range(1, len(set_words) + 1)
        tags = ['x']

        for _, memberPre in TagPrefix.__members__.items():
            for _, memberSuf in TagSurfix.__members__.items():
                if memberSuf is TagSurfix.S and memberPre is TagPrefix.general:
                    tags.append(memberPre.value + memberSuf.value)
                elif memberSuf != TagSurfix.S:
                    tags.append(memberPre.value + memberSuf.value)

        tags = list(set(tags))
        print(tags)

        tag_ids = range(len(tags))

        self.word2id = pd.Series(set_ids, index=set_words)
        self.id2word = pd.Series(set_words, index=set_ids)
        self.id2word[len(set_ids) + 1] = '<NEW>'
        self.word2id['<NEW>'] = len(set_ids) + 1

        self.tag2id = pd.Series(tag_ids, index=tags)
        self.id2tag = pd.Series(tags, index=tag_ids)

        self.df_data['X'] = self.df_data['words'].apply(self.X_padding)
        self.df_data['y'] = self.df_data['tags'].apply(self.y_padding)

        self.X = np.asarray(list(self.df_data['X'].values))
        self.y = np.asarray(list(self.df_data['y'].values))
        print('X.shape={}, y.shape={}'.format(self.X.shape, self.y.shape))
        print('Example of words: ', self.df_data['words'].values[0])
        print('Example of X: ', self.X[0])
        print('Example of tags: ', self.df_data['tags'].values[0])
        print('Example of y: ', self.y[0])

        with open(self.save_path, 'wb') as outp:
            pickle.dump(self.X, outp)
            pickle.dump(self.y, outp)
            pickle.dump(self.word2id, outp)
            pickle.dump(self.id2word, outp)
            pickle.dump(self.tag2id, outp)
            pickle.dump(self.id2tag, outp)
        print('** Finished saving the data.')

    def X_padding(self, words):

        ids = list(self.word2id[words])
        if len(ids) >= self.max_len:
            return ids[:self.max_len]
        ids.extend([0] * (self.max_len - len(ids)))
        return ids

    def y_padding(self, tags):
        ids = list(self.tag2id[tags])
        if len(ids) >= self.max_len:
            return ids[:self.max_len]
        ids.extend([0] * (self.max_len - len(ids)))
        return ids

    def builderTrainData(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        print(
            'X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
                X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))

        print('Creating the data generator ...')
        data_train = BatchGenerator(X_train, y_train, shuffle=True)
        data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
        data_test = BatchGenerator(X_test, y_test, shuffle=False)
        print('Finished creating the data generator.')

        return data_train, data_valid, data_test


class BatchGenerator(object):

    def __init__(self, X, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]


if __name__ == '__main__':
    data = DataHandler(rootDir='./corpus/2014', save_path='data/data_ner_0409.pkl')
    data.loadData()

    data.builderTrainData()
    print(data.X)
    print(type(data.X))
    print(data.X.shape)