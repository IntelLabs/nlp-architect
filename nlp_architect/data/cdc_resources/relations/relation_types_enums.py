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

from enum import Enum


class EmbeddingMethod(Enum):
    GLOVE = 'glove'
    GLOVE_OFFLINE = 'glove_offline'
    ELMO = 'elmo'
    ELMO_OFFLINE = 'elmo_offline'


class WikipediaSearchMethod(Enum):
    ONLINE = 'online'
    OFFLINE = 'offline'
    ELASTIC = 'elastic'


class OnlineOROfflineMethod(Enum):
    ONLINE = 'online'
    OFFLINE = 'offline'


class RelationType(Enum):
    NO_RELATION_FOUND = 0
    WIKIPEDIA_REDIRECT_LINK = 1
    WIKIPEDIA_ALIASES = 2
    WIKIPEDIA_DISAMBIGUATION = 3
    WIKIPEDIA_PART_OF_SAME_NAME = 4
    WIKIPEDIA_CATEGORY = 5
    WIKIPEDIA_TITLE_PARENTHESIS = 6
    WIKIPEDIA_BE_COMP = 7
    EXACT_STRING = 8
    FUZZY_FIT = 9
    FUZZY_HEAD_FIT = 10
    SAME_HEAD_LEMMA = 11
    VERBOCEAN_MATCH = 13
    WORDNET_DERIVATIONALLY = 14
    WORDNET_PARTIAL_SYNSET_MATCH = 15
    WORDNET_SAME_SYNSET = 17
    REFERENT_DICT = 18
    WORD_EMBEDDING_MATCH = 19
    WITHIN_DOC_COREF = 20
    OTHER = 21
