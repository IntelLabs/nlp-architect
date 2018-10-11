# -*- coding: utf-8 -*-

"""
Created on 2018/10/10 上午10:18

@author: xujiang@baixing.com

"""

import re
import numpy as np
import copy
import pickle
from collections import defaultdict

def batch_generator(arr, n_seqs, n_steps):
    arr = copy.copy(arr)
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    while True:
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y

class Tokenizer(object):
    def __init__(self, text=None, num_words=5000, vocab_path=None):
        if vocab_path is not None:
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            print('Vocabulary is loaded successfully.')
        else:
            # calculate word frequency
            word_count = defaultdict(int)
            for sentence in text:
                for word in sentence:
                    word_count[word] += 1
            vocab = list(word_count.keys())
            print(len(vocab), 'different characters')

            word_count_list = [(word, word_count[word]) for word in word_count]
            word_count_list.sort(key=lambda x: x[1], reverse=True)

            if len(word_count_list) > num_words:
                word_count_list = word_count_list[:num_words]

            vocab = [x[0] for x in word_count_list]
            self.vocab = vocab

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_num(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def num_to_word(self, index):
        if index == len(self.vocab):
            return '还'
            # return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Index out of range!')

    def texts_to_sequences(self, text):
        sequences = []
        for sentence in text:
            for word in sentence:
                sequences.append(self.word_to_num(word))
        return np.array(sequences)

    def sequences_to_texts(self, sequences):
        texts = [self.num_to_word(index) for index in sequences]
        text =  "".join(texts)
        texts = re.split('！|。|？', text, 1)
        print (texts)
        text = texts[1] if len(texts)>1 else texts[0]
        print (text)
        return text.split('E')[0]

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)