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

import logging
import os
from typing import Dict, List, Set

from nlp_architect.common.cdc.mention_data import MentionDataLight
from nlp_architect.data.cdc_resources.relations.relation_extraction import RelationExtraction
from nlp_architect.data.cdc_resources.relations.relation_types_enums import \
    OnlineOROfflineMethod, RelationType
from nlp_architect.utils.io import load_json_file
from nlp_architect.utils.string_utils import StringUtils

logger = logging.getLogger(__name__)


class ReferentDictRelationExtraction(RelationExtraction):
    def __init__(self, method: OnlineOROfflineMethod = OnlineOROfflineMethod.ONLINE,
                 ref_dict: str = None):
        """
        Extract Relation between two mentions according to Referent Dictionary knowledge

        Args:
            method (optional): OnlineOROfflineMethod.{ONLINE/OFFLINE} run against full referent
                dictionary or a sub-set of (default = ONLINE)
            ref_dict (required): str Location of referent dictionary file to work with
        """
        logger.info('Loading ReferentDict module')
        if ref_dict is not None and os.path.isfile(ref_dict):
            if method == OnlineOROfflineMethod.OFFLINE:
                self.ref_dict = load_json_file(ref_dict)
            elif method == OnlineOROfflineMethod.ONLINE:
                self.ref_dict = self.load_reference_dict(ref_dict)
            logger.info('ReferentDict module lead successfully')
        else:
            raise FileNotFoundError('Referent Dict file not found or not in path:' + ref_dict)

        super(ReferentDictRelationExtraction, self).__init__()

    def extract_all_relations(self, mention_x: MentionDataLight,
                              mention_y: MentionDataLight) -> Set[RelationType]:
        ret_ = set()
        ret_.add(self.extract_sub_relations(mention_x, mention_y, RelationType.REFERENT_DICT))
        return ret_

    def extract_sub_relations(self, mention_x: MentionDataLight, mention_y: MentionDataLight,
                              relation: RelationType) -> RelationType:
        """
        Check if input mentions has the given relation between them

        Args:
            mention_x: MentionDataLight
            mention_y: MentionDataLight
            relation: RelationType

        Returns:
            RelationType: relation in case mentions has given relation or
                RelationType.NO_RELATION_FOUND otherwise
        """
        if relation is not RelationType.REFERENT_DICT:
            return RelationType.NO_RELATION_FOUND

        mention_x_str = mention_x.tokens_str
        mention_y_str = mention_y.tokens_str
        if StringUtils.is_pronoun(mention_x_str.lower()) or StringUtils.is_pronoun(
                mention_y_str.lower()):
            return RelationType.NO_RELATION_FOUND

        if self.is_referent_dict(mention_x, mention_y):
            return RelationType.REFERENT_DICT

        return RelationType.NO_RELATION_FOUND

    def is_referent_dict(self, mention_x: MentionDataLight, mention_y: MentionDataLight) -> bool:
        """
        Check if input mentions has referent dictionary relation between them

        Args:
            mention_x: MentionDataLight
            mention_y: MentionDataLight

        Returns:
            bool
        """
        match_result = False
        x_head = mention_x.mention_head
        y_head = mention_y.mention_head
        x_head_lemma = mention_x.mention_head_lemma
        y_head_lemma = mention_y.mention_head_lemma

        if (x_head in self.ref_dict and y_head in self.ref_dict[x_head]) or (
                y_head in self.ref_dict and x_head in self.ref_dict[y_head]):
            match_result = True
        if (x_head_lemma in self.ref_dict and y_head_lemma in self.ref_dict[x_head_lemma]) or \
                (y_head_lemma in self.ref_dict and x_head_lemma in self.ref_dict[y_head_lemma]):
            match_result = True
        if (x_head_lemma in self.ref_dict and y_head in self.ref_dict[x_head_lemma]) or (
                y_head_lemma in self.ref_dict and x_head in self.ref_dict[y_head_lemma]):
            match_result = True
        if (y_head in self.ref_dict and x_head_lemma in self.ref_dict[y_head]) or \
                (x_head in self.ref_dict and y_head_lemma in self.ref_dict[x_head]):
            match_result = True

        return match_result

    @staticmethod
    def get_supported_relations():
        """
        Return all supported relations by this class

        Returns:
            List[RelationType]
        """
        return [RelationType.REFERENT_DICT]

    @staticmethod
    def load_reference_dict(dict_fname: str) -> Dict[str, List[str]]:
        """
        Method to load referent dictionary to memory

        Returns:
            List[RelationType]
        """
        word_dict = {}
        first = True
        with open(dict_fname, 'r', encoding="utf-8") as f:
            for line in f:
                if first:
                    first = False
                    continue
                word1, word2, _, npmi = line.strip().split('\t')
                npmi = float(npmi)
                if npmi >= 0.2:
                    if word1 not in word_dict:
                        word_dict[word1] = []
                    word_dict[word1].append(word2)
        return word_dict
