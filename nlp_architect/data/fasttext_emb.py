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
import os
from six.moves import urllib
import numpy as np
from nlp_architect.utils.generic import license_prompt


class FastTextEmb:
    """
    Downloads FastText Embeddings for a given language to the given path.
    Arguments:
        path(str): Local path to copy embeddings
        language(str): Embeddings language
        vocab_size(int): Size of vocabulary
    Returns:
        Returns a dictionary and reverse dictionary
        Returns a numpy array with embeddings in emb_sizexvocab_size shape
    """

    def __init__(self, path, language, vocab_size, emb_dim=300):
        self.path = path
        self.language = language
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.url = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki." + language + ".vec"

    def _maybe_download(self):
        """
        Download filename from url unless it's already in directory
        """
        # 1. Check if the file doesnt exist. Download and extract if it doesnt
        filename = "wiki." + self.language + ".vec"
        filepath = os.path.join(self.path, filename)
        link = "https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md"
        if not os.path.exists(filepath):
            if license_prompt(filepath, link, self.path):
                print(
                    "Downloading FastText embeddings for " + self.language + " to " + filepath)
                urllib.request.urlretrieve(self.url, filepath)
                statinfo = os.stat(filepath)
                print(
                    "Sucessfully downloaded", filename, statinfo.st_size, "bytes")
            else:
                exit()
        else:
            print(
                "Found FastText embeddings for " + self.language + " at " + filepath)
        return filepath

    def read_embeddings(self, filepath):
        word2id = {}
        word_vec = []
        with open(filepath) as emb_file:
            for i, line in enumerate(emb_file):
                # Line zero has total words, emb dimensions
                if i == 0:
                    split_line = line.split()
                    assert len(split_line) == 2
                    assert self.emb_dim == int(split_line[1])
                # Rest of line are word, word_vec format
                else:
                    word, vector = line.rstrip().split(' ', 1)
                    vector = np.fromstring(vector, sep=' ')
                    # If norm is zero fill with 0.01
                    if np.linalg.norm(vector) == 0:
                        vector[0] = 0.01
                    assert vector.shape == (self.emb_dim, ), i
                    # Assign a token
                    word2id[word] = len(word2id)
                    word_vec.append(vector[None])
                # Check if your reached goal of vocab_size
                if i >= self.vocab_size:
                    break
        # Reverse dictionary
        id2word = {v: k for k, v in word2id.items()}
        # Dictionary just combines both id2word and word2id into one dict
        dico = Dictionary(id2word, word2id, self.language)
        # All word_vectors
        word_vec = np.concatenate(word_vec, 0)
        # Normalize the embeddings
        return dico, word_vec

    def load_embeddings(self):
        # Check if embeddings exist else download
        filepath = self._maybe_download()
        # Read embeddings
        dico, word_vec = self.read_embeddings(filepath)
        print("Completed loading embeddings for " + self.language)
        word_vec = np.float32(word_vec)
        return dico, word_vec


def get_eval_data(eval_path, src_lang, tgt_lang):
    """
    Downloads evaluation cross lingual dictionaries to the eval_path
    Arguments:
        eval_path: Path where cross-lingual dictionaries are downloaded
        src_lang : Source Language
        tgt_lang : Target Language
    Returns:
        Path to where cross lingual dictionaries are downloaded
    """
    eval_url = 'https://s3.amazonaws.com/arrival/dictionaries/'
    link = "https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries"
    src_path = os.path.join(eval_path, '%s-%s.5000-6500.txt' % (src_lang, tgt_lang))
    filename = src_lang + '-' + tgt_lang + '.5000-6500.txt'
    if not os.path.exists(src_path):
        if license_prompt(src_path, link, src_path):
            os.system("mkdir -p " + eval_path)
            print("Downloading cross-lingual dictionaries for " + src_lang)
            urllib.request.urlretrieve(eval_url + filename, src_path)
            print("Completed downloading to " + eval_path)
        else:
            exit()
    return src_path


class Dictionary:
    """
    Merges word2idx and idx2word dictionaries
    Arguments:
        id2word dictionary
        word2id dictionary
        language of the dictionary
    Usage:
        dico.index(word) - returns an index
        dico[index] - returns the word
    """

    def __init__(self, id2word, word2id, lang):
        assert len(id2word) == len(word2id)
        self.id2word = id2word
        self.word2id = word2id
        self.lang = lang
        self.check_valid()

    def __len__(self):
        """
        Returns the number of words in the dictionary.
        """
        return len(self.id2word)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.id2word[i]

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return w in self.word2id

    def __eq__(self, y):
        """
        Compare the dictionary with another one.
        """
        self.check_valid()
        y.check_valid()
        if len(self.id2word) != len(y):
            return False
        return self.lang == y.lang and all(
            self.id2word[i] == y[i] for i in range(len(y)))

    def check_valid(self):
        """
        Check that the dictionary is valid.
        """
        assert len(self.id2word) == len(self.word2id)
        for i in range(len(self.id2word)):
            assert self.word2id[self.id2word[i]] == i

    def index(self, word):
        """
        Returns the index of the specified word.
        """
        return self.word2id[word]
