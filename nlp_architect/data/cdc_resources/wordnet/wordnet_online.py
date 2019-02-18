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
from nltk.corpus import wordnet as wn

from nlp_architect.data.cdc_resources.data_types.wn.wordnet_page import WordnetPage
from nlp_architect.utils.string_utils import StringUtils


class WordnetOnline(object):
    def __init__(self):
        self.cache = dict()
        nltk.download('wordnet')

    def get_pages(self, mention):
        if mention.tokens_str in self.cache:
            return self.cache[mention.tokens_str]

        head_synonyms, head_names_derivationally = self.extract_synonyms_and_derivation(
            mention.mention_head)
        head_lemma_synonyms, head_lemma_derivationally = self.extract_synonyms_and_derivation(
            mention.mention_head_lemma)
        clean_phrase = StringUtils.normalize_str(mention.tokens_str)
        all_clean_words_synonyms = self.all_clean_words_synonyms(clean_phrase)

        wordnet_page = WordnetPage(mention.tokens_str, clean_phrase, mention.mention_head,
                                   mention.mention_head_lemma,
                                   head_synonyms,
                                   head_lemma_synonyms, head_names_derivationally,
                                   head_lemma_derivationally,
                                   all_clean_words_synonyms)

        self.cache[mention.tokens_str] = wordnet_page
        return wordnet_page

    @staticmethod
    def extract_synonyms_and_derivation(word):
        lemma_names = set()
        derivationally_related_forms = set()
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                lemma_name = lemma.name().replace('_', ' ')
                if not StringUtils.is_stop(lemma_name.lower()):
                    lemma_names.add(lemma_name)

                derivationally_related_forms.update(
                    [l.name().replace('_', ' ') for l in lemma.derivationally_related_forms()
                     if not StringUtils.is_stop(l.name().lower())])

        return lemma_names, derivationally_related_forms

    @staticmethod
    def all_clean_words_synonyms(clean_phrase):
        words = clean_phrase.split()
        return [set([lemma.lower().replace('_', ' ')
                     for synset in wn.synsets(w)
                     for lemma in synset.lemma_names() if not StringUtils.is_stop(lemma.lower())])
                for w in words]
