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
import re
import sys
from os import path
from typing import List, Tuple

import spacy
from nltk import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from spacy.cli.download import download as spacy_download
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from spacy.lemmatizer import Lemmatizer

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


def try_to_load_spacy(model_name):
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
        display_prompt (bool, optional): flag to display/skip license prompt
    """

    def __init__(self, model='en', disable=None, display_prompt=True):
        if disable is None:
            disable = []
        try:
            self._parser = spacy.load(model, disable=disable)
        except OSError:
            url = 'https://spacy.io/models'
            if display_prompt and license_prompt('Spacy {} model'.format(model), url) is False:
                sys.exit(0)
            spacy_download(model)
            self._parser = spacy.load(model, disable=disable)

    @property
    def parser(self):
        """return Spacy's instance parser"""
        return self._parser

    def tokenize(self, text: str) -> List[str]:
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


class Stopwords(object):
    """
    Stop words list class.
    """
    stop_words = []

    @staticmethod
    def get_words():
        if not Stopwords.stop_words:
            sw_path = path.join(path.dirname(path.realpath(__file__)),
                                'resources',
                                'stopwords.txt')
            with open(sw_path) as fp:
                stop_words = []
                for w in fp:
                    stop_words.append(w.strip().lower())
            Stopwords.stop_words = stop_words
        return Stopwords.stop_words


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
        2D numpy array and Vocabulary of the detected words
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


def bio_to_spans(text: List[str], tags: List[str]) -> List[Tuple[int, int, str]]:
    """
    Convert BIO tagged list of strings into span starts and ends
    Args:
        text: list of words
        tags: list of tags

    Returns:
        tuple: list of start, end and tag of detected spans
    """
    pointer = 0
    starts = []
    for i, t, in enumerate(tags):
        if t.startswith('B-'):
            starts.append((i, pointer))
        pointer += len(text[i]) + 1

    spans = []
    for s_i, s_char in starts:
        label_str = tags[s_i][2:]
        e = 0
        e_char = len(text[s_i + e])
        while len(tags) > s_i + e + 1 and tags[s_i + e + 1].startswith('I-'):
            e += 1
            e_char += 1 + len(text[s_i + e])
        spans.append((s_char, s_char + e_char, label_str))
    return spans
