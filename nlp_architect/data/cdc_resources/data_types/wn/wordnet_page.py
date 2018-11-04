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
from typing import Set, Dict


class WordnetPage(object):
    def __init__(self, orig_phrase: str, clean_phrase: str, head: str, head_lemma: str,
                 head_synonyms: Set[str], head_lemma_synonyms: Set[str],
                 head_derivationally: Set[str], head_lemma_derivationally: Set[str],
                 all_clean_words_synonyms: Set[str]) -> None:
        """
        Object represent a Wikipedia Page and extracted fields.

        Args:
            orig_phrase (str): original search phrase
            clean_phrase (str): original search phrase normalized
            head (str): page title head
            head_lemma (str): page title head lemma
            head_synonyms (set): head synonyms words extracted from wordnet
            head_lemma_synonyms (set): head lemma synonyms words extracted from wordnet
            head_derivationally (set): wordnet head derivationally_related_forms()
            head_lemma_derivationally (set): wordnet head lemma derivationally_related_forms()
            all_clean_words_synonyms (set): clean_phrase wordnet synonyms
        """
        self.orig_phrase = orig_phrase
        self.clean_phrase = clean_phrase
        self.head = head
        self.head_lemma = head_lemma
        self.head_synonyms = head_synonyms
        self.head_lemma_synonyms = head_lemma_synonyms
        self.head_derivationally = head_derivationally
        self.head_lemma_derivationally = head_lemma_derivationally
        self.all_clean_words_synonyms = all_clean_words_synonyms

    def __eq__(self, other):
        return self.orig_phrase == other.orig_phrase and self.head == other.head and \
            self.head_lemma == other.head_lemma

    def __hash__(self):
        return hash(self.orig_phrase) + hash(self.head) + hash(self.head_lemma)

    def toJson(self) -> Dict:
        result_dict = dict()
        result_dict['orig_phrase'] = self.orig_phrase
        result_dict['clean_phrase'] = self.clean_phrase
        result_dict['head'] = self.head
        result_dict['head_lemma'] = self.head_lemma
        result_dict['head_synonyms'] = list(self.head_synonyms)
        result_dict['head_lemma_synonyms'] = list(self.head_lemma_synonyms)
        result_dict['head_derivationally'] = list(self.head_derivationally)
        result_dict['head_lemma_derivationally'] = list(self.head_lemma_derivationally)
        if self.all_clean_words_synonyms is not None:
            all_as_list = []
            for set_ in self.all_clean_words_synonyms:
                all_as_list.append(list(set_))
            result_dict['all_clean_words_synonyms'] = all_as_list

        return result_dict
