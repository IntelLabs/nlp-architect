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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import spacy
from spacy.cli.download import download as spacy_download


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

    def __len__(self):
        return len(self._vocab)

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


class SpacyPipeline:
    """
    Spacy pipeline wrapper which prompts user for model download authorization.

    Args:
        model (str, optional): spacy model name (default: english small model)
        disable (list of string, optional): pipeline annotators to disable
            (default: ['tagger', 'ner', 'parser', 'vectors', 'textcat'])
    """

    def __init__(self, model='en', **kwargs):
        if 'disable' not in kwargs:
            kwargs['disable'] = ['tagger', 'ner', 'parser', 'vectors', 'textcat']

        try:
            self.parser = spacy.load(model, **kwargs)
        except OSError:
            print('Spacy English model was not found')
            url = 'https://spacy.io/models/en#en_core_web_sm'
            print('License: Creative Commons v3-BY-SA '
                  'https://creativecommons.org/licenses/by-sa/3.0/')
            response = input('To download the model from {}, '
                             + 'please type YES: '.format(url))
            if response.lower().strip() == "yes":
                print('The terms and conditions of the data set license apply. Intel does not '
                      'grant any rights to the data files or database')
                print('Downloading Spacy model...')
                spacy_download(model)
                self.parser = spacy.load(model, **kwargs)
            else:
                print('Download declined. Response received {} != YES. '.format(response))
                print('Please download the model manually')
                sys.exit(0)

    def tokenize(self, text):
        """
        Tokenize a sentence into tokens
        Args:
            text (str): text to tokenize

        Returns:
            list: a list of str tokens of input
        """
        return [t.text for t in self.parser(text)]
