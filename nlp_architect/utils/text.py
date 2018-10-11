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
import copy
import sys
import re
import pickle
import numpy as np
from collections import defaultdict


import spacy
from spacy.cli.download import download as spacy_download
from nltk.stem.snowball import EnglishStemmer
from nltk import WordNetLemmatizer
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

from nlp_architect.utils.generic import license_prompt


class Vocabulary:
    """
    A vocabulary that maps words to ints (storing a vocabulary)
    """

    def __init__(self, start=0):

        self._vocab = {}
        self._rev_vocab = {}
        self.next = start

    def add(self, word):
        """
        Add word to vocabulary

        Args:
            word (str): word to add

        Returns:
            int: id of added word
        """
        if word not in self._vocab.keys():
            self._vocab[word] = self.next
            self._rev_vocab[self.next] = word
            self.next += 1
        return self._vocab.get(word)

    def word_id(self, word):
        """
        Get the word_id of given word

        Args:
            word (str): word from vocabulary

        Returns:
            int: int id of word
        """
        return self._vocab.get(word, None)

    def __getitem__(self, item):
        """
        Get the word_id of given word (same as `word_id`)
        """
        return self.word_id(item)

    def __len__(self):
        return len(self._vocab)

    def __iter__(self):
        for word in self.vocab.keys():
            yield word

    @property
    def max(self):
        return self.next

    def id_to_word(self, wid):
        """
        Word-id to word (string)

        Args:
            wid (int): word id

        Returns:
            str: string of given word id
        """
        return self._rev_vocab.get(wid)

    @property
    def vocab(self):
        """
        dict: get the dict object of the vocabulary
        """
        return self._vocab

    def add_vocab_offset(self, offset):
        """
        Adds an offset to the ints of the vocabulary

        Args:
            offset (int): an int offset
        """
        new_vocab = {}
        for k, v in self.vocab.items():
            new_vocab[k] = v + offset
        self.next += offset
        self._vocab = new_vocab
        self._rev_vocab = {v: k for k, v in new_vocab.items()}

    def reverse_vocab(self):
        """
        Return the vocabulary as a reversed dict object

        Returns:
            dict: reversed vocabulary object
        """
        return self._rev_vocab

class Tokenizer(object):
    def __init__(self, text=None, num_words=5000, vocab_path=None):
        if vocab_path is not None:
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            print('Vocabulary is loaded successfully.')
        else:
            # calculate word frequency
            word_count = defaultdict(int)
            for word in text:
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
        sequences = [self.word_to_num(word) for word in text]
        return np.array(sequences)

    def sequences_to_texts(self, sequences):
        texts = [self.num_to_word(index) for index in sequences]
        return "".join(texts)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)

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

def is_spacy_model_installed(model_name):
    try:
        spacy.load(model_name)
        return True
    except OSError:
        return False


class SpacyInstance:
    """
    Spacy pipeline wrapper which prompts user for model download authorization.

    Args:
        model (str, optional): spacy model name (default: english small model)
        disable (list of string, optional): pipeline annotators to disable
            (default: [])
    """

    def __init__(self, model='en_core_web_sm', disable=None):
        if disable is None:
            disable = []
        try:
            self._parser = spacy.load(model, disable=disable)
        except OSError:
            url = 'https://spacy.io/models'
            if license_prompt('Spacy {} model'.format(model), url) is False:
                sys.exit(0)
            spacy_download(model)
            self._parser = spacy.load(model, disable=disable)

    @property
    def parser(self):
        """return Spacy's instance parser"""
        return self._parser

    def tokenize(self, text):
        """
        Tokenize a sentence into tokens
        Args:
            text (str): text to tokenize

        Returns:
            list: a list of str tokens of input
        """
        # pylint: disable=not-callable

        return [t.text for t in self.parser(text)]


stemmer = EnglishStemmer()
lemmatizer = WordNetLemmatizer()
spacy_lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
p = re.compile(r'[ \-,;.@&_]')


def simple_normalizer(text):
    """
    Simple text normalizer. Runs each token of a phrase thru wordnet lemmatizer
    and a stemmer.
    """
    if not str(text).isupper() or \
            not str(text).endswith('S') or \
            not len(text.split()) == 1:
        tokens = list(filter(lambda x: len(x) != 0, p.split(text.strip())))
        text = ' '.join([stemmer.stem(lemmatizer.lemmatize(t))
                         for t in tokens])
    return text


def spacy_normalizer(text, lemma=None):
    """
    Simple text normalizer using spacy lemmatizer. Runs each token of a phrase
    thru a lemmatizer and a stemmer.
    Arguments:
        text(string): the text to normalize.
        lemma(string): lemma of the given text. in this case only stemmer will
        run.
    """
    if not str(text).isupper() or \
            not str(text).endswith('S') or \
            not len(text.split()) == 1:
        tokens = list(filter(lambda x: len(x) != 0, p.split(text.strip())))
        if lemma:
            lemma = lemma.split(' ')
            text = ' '.join([stemmer.stem(l)
                             for l in lemma])
        else:
            text = ' '.join([stemmer.stem(spacy_lemmatizer(t, u'NOUN')[0])
                             for t in tokens])
    return text


def read_sequential_tagging_file(file_path, ignore_line_patterns=None):
    """
    Read a tab separated sequential tagging file.
    Returns a list of list of tuple of tags (sentences, words)

    Args:
        file_path (str): input file path
        ignore_line_patterns (list, optional): list of string patterns to ignore

    Returns:
        list of list of tuples
    """
    if ignore_line_patterns:
        assert isinstance(ignore_line_patterns, list), 'ignore_line_patterns must be a list'

    def _split_into_sentences(file_lines):
        sentences = []
        s = []
        for line in file_lines:
            if len(line) == 0:
                sentences.append(s)
                s = []
                continue
            s.append(line)
        if len(s) > 0:
            sentences.append(s)
        return sentences

    with open(file_path, encoding='utf-8') as fp:
        data = fp.readlines()
        data = [d.strip() for d in data]
        if ignore_line_patterns:
            for s in ignore_line_patterns:
                data = [d for d in data if s not in d]
        data = [tuple(d.split()) for d in data]
    return _split_into_sentences(data)


def word_vector_generator(data, lower=False, start=0):
    """
    Word vector generator util.
    Transforms a list of sentences into numpy int vectors and returns the
    constructed vocabulary

    Arguments:
        data (list): list of list of strings
        lower (bool, optional): transform strings into lower case
        start (int, optional): vocabulary index start integer

    Returns:
        np.array: a 2D numpy array
        Vocabulary: constructed vocabulary
    """
    vocab = Vocabulary(start)
    data_vec = []
    for sentence in data:
        sentence_vec = []
        for w in sentence:
            word = w
            if lower:
                word = word.lower()
            wid = vocab[word]
            if wid is None:
                wid = vocab.add(word)
            sentence_vec.append(wid)
        data_vec.append(sentence_vec)
    return data_vec, vocab


def character_vector_generator(data, start=0):
    """
    Character word vector generator util.
    Transforms a list of sentences into numpy int vectors of the characters
    of the words of the sentence, and returns the constructed vocabulary

    Arguments:
        data (list): list of list of strings
        start (int, optional): vocabulary index start integer

    Returns:
        np.array: a 2D numpy array
        Vocabulary: constructed vocabulary
    """
    vocab = Vocabulary(start)
    data_vec = []
    for sentence in data:
        sentence_vec = []
        for w in sentence:
            word_vec = []
            for char in w:
                cid = vocab[char]
                if cid is None:
                    cid = vocab.add(char)
                word_vec.append(cid)
            sentence_vec.append(word_vec)
        data_vec.append(sentence_vec)
    return data_vec, vocab


def extract_nps(annotation_list, text=None):
    """
    Extract Noun Phrases from given text tokens and phrase annotations.
    Returns a list of tuples with start/end indexes.

    Args:
        annotation_list (list): a list of annotation tags in str
        text (list, optional): a list of token texts in str

    Returns:
        list of start/end markers of noun phrases, if text is provided a list of noun phrase texts
    """
    np_starts = [i for i in range(len(annotation_list)) if annotation_list[i] == 'B-NP']
    np_markers = []
    for s in np_starts:
        i = 1
        while s + i < len(annotation_list) and annotation_list[s + i] == 'I-NP':
            i += 1
        np_markers.append((s, s + i))
    return_markers = np_markers
    if text:
        assert len(text) == len(annotation_list), 'annotations/text length mismatch'
        return_markers = [' '.join(text[s:e]) for s, e in np_markers]
    return return_markers
