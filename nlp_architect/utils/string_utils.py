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
import re
import string
from typing import List

from nlp_architect.utils.io import load_json_file
from nlp_architect.utils.text import SpacyInstance

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

STOP_WORDS_FILE = os.path.join(CURRENT_DIR, 'resources/stop_words_en.json')
PRONOUN_FILE = os.path.join(CURRENT_DIR, 'resources/pronoun_en.json')
PREPOSITION_FILE = os.path.join(CURRENT_DIR, 'resources/preposition_en.json')
DETERMINERS_FILE = os.path.join(CURRENT_DIR, 'resources/determiners_en.json')

DISAMBIGUATION_CATEGORY = ['disambig', 'disambiguation']


class StringUtils:
    spacy_no_parser = SpacyInstance(disable=['parser'])
    spacy_parser = SpacyInstance()
    stop_words = []
    pronouns = []
    preposition = []
    determiners = []

    def __init__(self):
        pass

    @staticmethod
    def is_stop(token: str) -> bool:
        if not StringUtils.stop_words:
            StringUtils.stop_words = load_json_file(STOP_WORDS_FILE)
            StringUtils.stop_words.extend(DISAMBIGUATION_CATEGORY)
        if token not in StringUtils.stop_words:
            return False
        return True

    @staticmethod
    def normalize_str(in_str: str) -> str:
        str_clean = re.sub('[' + string.punctuation + string.whitespace + ']', ' ',
                           in_str).strip().lower()
        if isinstance(str_clean, str):
            str_clean = str(str_clean)

        parser = StringUtils.spacy_no_parser.parser
        doc = parser(str_clean)
        ret_clean = []
        for token in doc:
            lemma = token.lemma_.strip()
            if not StringUtils.is_pronoun(lemma) and not StringUtils.is_stop(lemma):
                ret_clean.append(token.lemma_)

        return ' '.join(ret_clean)

    @staticmethod
    def is_pronoun(in_str: str) -> bool:
        if not StringUtils.pronouns:
            StringUtils.pronouns = load_json_file(PRONOUN_FILE)

        tokens = in_str.split()
        if len(tokens) == 1:
            if tokens[0] in StringUtils.pronouns:
                return True
        return False

    @staticmethod
    def is_determiner(in_str: str) -> bool:
        if not StringUtils.determiners:
            StringUtils.determiners = load_json_file(DETERMINERS_FILE)

        tokens = in_str.split()
        if len(tokens) == 1:
            if tokens[0] in StringUtils.determiners:
                return True
        return False

    @staticmethod
    def is_preposition(in_str: str) -> bool:
        if not StringUtils.preposition:
            StringUtils.preposition = load_json_file(PREPOSITION_FILE)

        tokens = in_str.split()
        if len(tokens) == 1:
            if tokens[0] in StringUtils.preposition:
                return True
        return False

    @staticmethod
    def normalize_string_list(str_list: str) -> List[str]:
        ret_list = []
        for _str in str_list:
            normalize_str = StringUtils.normalize_str(_str)
            if normalize_str != '':
                ret_list.append(normalize_str)
        return ret_list

    @staticmethod
    def find_head_lemma_pos_ner(x: str):
        """"

        :param x: mention
        :return: the head word and the head word lemma of the mention
        """
        head = "UNK"
        lemma = "UNK"
        pos = "UNK"
        ner = "UNK"

        # pylint: disable=not-callable
        doc = StringUtils.spacy_parser.parser(x)
        for tok in doc:
            if tok.head == tok:
                head = tok.text
                lemma = tok.lemma_
                pos = tok.pos_

        for ent in doc.ents:
            if ent.root.text == head:
                ner = ent.label_

        return head, lemma, pos, ner
