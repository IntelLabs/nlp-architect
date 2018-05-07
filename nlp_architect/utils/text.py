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


class Vocabulary:
    """
    A vocabulary that maps words to ints (storing a vocabulary)
    """
    def __init__(self):
        self._vocab = {}
        self._rev_vocab = {}
        self.next = 0

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
