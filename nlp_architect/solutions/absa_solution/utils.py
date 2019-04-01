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
import csv
import re
from abc import abstractmethod


class Anonymiser(object):
    """Abstract class for anonymiser algorithm, intended for privacy keeping."""
    @abstractmethod
    def run(self, text):
        pass


class TweetAnonymiser(Anonymiser):
    """Anonymiser for tweets which uses lexicon for simple string replacements."""
    def __init__(self, lexicon_path):
        self.entity_dict = self._init_entity_dict(lexicon_path)

    @staticmethod
    def _init_entity_dict(lexicon_path):
        ret = {}
        with open(lexicon_path) as f:
            for row in csv.reader(f):
                ret[row[0]] = [_ for _ in row[1:] if _]
        return ret

    def run(self, text):
        for anonymised, entities in self.entity_dict.items():
            for entity in entities:
                text = re.sub(entity, anonymised, text, flags=re.IGNORECASE)
        text = ' '.join(["@other_entity" if (word.startswith('@') and word[1:]
                                             not in self.entity_dict.keys())
                         else word for word in text.split()])
        return text
