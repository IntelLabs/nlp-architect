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

import nltk
import nltk.corpus
import numpy as np
import requests
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import wordnet as wn
from nltk.stem.snowball import SnowballStemmer

from nlp_architect.utils.generic import license_prompt

stemmer = SnowballStemmer("english")
headers = {"Accept": "application/json"}


def stem(w):
    """
    Stem input

    Args:
        w (str): word to extract stem

    Returns:
        str: stem of w
    """
    return stemmer.stem(w)


class Wikidata:
    """
    Wikidata service

    Args:
        http_proxy(str) : http proxy
        https_proxy(str) : https proxy
    """

    def __init__(self, http_proxy=None, https_proxy=None):
        self.headers = headers
        proxies = {}
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy
        self.proxies = proxies

    def find_wikidata_existence(self, candidates):
        """
        extract Wikidata indicator-feature (1 if exist in Wikidata, else 0)

        Args:
            candidates(list(str)): a list of all possible candidates to have Wikidata entry

        Returns :
            int: 1 if exist in Wikidata for any candidate in candidates, else 0
        """
        for candidate in set(candidates):
            if self.has_item(candidate):
                return 1
        return 0

    def has_item(self, phrase):
        """
        Send a SPARQL query to wikidata, and return response

        Args:
            phrase (str):  a noun-phrase

        Returns:
            bool: True if exist in Wikidata for phrase, else False
        """
        chr_url = """https://query.wikidata.org/sparql?query=
                        SELECT ?item ?lable
                        WHERE
                        {
                            ?item ?label '""" + phrase + """'@en .
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
                        }
                        &format = JSON"""
        r = requests.get(chr_url, headers=self.headers, proxies=self.proxies)
        empty_result = b'{\n  "head" : {\n    "vars" : [ "item",' \
                       b' "lable" ]\n  },\n  ' \
                       b'"results" : {\n    "bindings" : [ ]\n  }\n}'
        if r.status_code == 200 and empty_result != r.content:
            return True
        return False


class Word2Vec:
    """
    Word2Vec service

        Args:
            word2vec_model_path (str): the local path to the Word2Vec pre-trained model
    """

    def __init__(self, word2vec_model_path):
        self.word2vec_model_path = word2vec_model_path
        self.model = self.load_word2vec_model_from_path()

    def load_word2vec_model_from_path(self):
        """
        Load Word2Vec model

        Returns:
            the Word2Vec model
        """
        word_embeddings_model = KeyedVectors.load_word2vec_format(
            self.word2vec_model_path, binary=True)
        if not word_embeddings_model:
            return None
        return word_embeddings_model

    def get_word_embedding(self, word):
        """
        Get the pre-trained word embeddings

        Args:
            word (str): the word to extract from the models embedding

        Returns:
            :obj:`np.ndarray`: the word embeddings
        """
        if not self.model:
            return np.full((300,), -1)
        if word in self.model.vocab:
            vec = np.array(self.model.word_vec(word))
        else:
            vec = np.full((300,), -1)
        return vec

    def get_similarity_score(self, noun_phrase):
        """
            Get the cosign similarity distance between the np words (only if 2)

        Args:
            noun_phrase (str): the noun-phrase

        Returns:
            float: cosign similarity distance between the np words (only if 2)
        """
        candidates = noun_phrase.split(" ")
        if len(candidates) < 2:
            candidates.extend(candidates[0])
        if len(candidates) > 2:
            return -1
        try:
            if not self.model:
                return -1
            return self.model.similarity(candidates[0], candidates[1])
        except KeyError:
            return -1


class Wordnet:
    """
    WordNet service
    """

    def __init__(self):
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            if license_prompt('WordNet data set', 'http://www.nltk.org/nltk_data/') is False:
                raise Exception("can't continue data prepare process "
                                "without downloading WordNet dataset")
            nltk.download('wordnet')
        self.wordnet = wn

    def find_wordnet_existence(self, candidates):
        """
        extract WordNet indicator-feature (1 if exist in WordNet, else 0)

        Args:
            candidates (list(str)): a list of all possible candidates to have WordNet entry

        Returns:
            int: 1 if exist in WordNet for any candidate in candidates, else 0
        """
        for candidate in candidates:
            candidate = candidate.replace(" ", "_")
            if self.wordnet.synsets(candidate):
                return 1
        return 0
