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
from typing import Dict, Set

from nlp_architect.common.cdc.mention_data import MentionDataLight
from nlp_architect.data.cdc_resources.relations.relation_extraction import RelationExtraction
from nlp_architect.data.cdc_resources.relations.relation_types_enums import RelationType, \
    OnlineOROfflineMethod
from nlp_architect.utils.io import load_json_file
from nlp_architect.utils.string_utils import StringUtils

logger = logging.getLogger(__name__)


class VerboceanRelationExtraction(RelationExtraction):
    def __init__(self, method: OnlineOROfflineMethod = OnlineOROfflineMethod.ONLINE,
                 vo_file: str = None):
        """
        Extract Relation between two mentions according to VerbOcean knowledge

        Args:
            method (optional): OnlineOROfflineMethod.{ONLINE/OFFLINE} run against full VerbOcean or
                a sub-set of it (default = ONLINE)
            vo_file (required): str Location of VerbOcean file to work with
        """
        logger.info('Loading Verb Ocean module')
        if vo_file is not None and os.path.isfile(vo_file):
            if method == OnlineOROfflineMethod.OFFLINE:
                self.vo = load_json_file(vo_file)
            elif method == OnlineOROfflineMethod.ONLINE:
                self.vo = self.load_verbocean_file(vo_file)
            logger.info('Verb Ocean module lead successfully')
        else:
            raise FileNotFoundError('VerbOcean file not found or not in path..')
        super(VerboceanRelationExtraction, self).__init__()

    def extract_all_relations(self, mention_x: MentionDataLight,
                              mention_y: MentionDataLight) -> Set[RelationType]:
        ret_ = set()
        ret_.add(self.extract_sub_relations(mention_x, mention_y, RelationType.VERBOCEAN_MATCH))
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
        if relation is not RelationType.VERBOCEAN_MATCH:
            return RelationType.NO_RELATION_FOUND

        mention_x_str = mention_x.tokens_str
        mention_y_str = mention_y.tokens_str
        if StringUtils.is_pronoun(mention_x_str.lower()) or StringUtils.is_pronoun(
                mention_y_str.lower()):
            return RelationType.NO_RELATION_FOUND

        if self.is_verbocean_relation(mention_x, mention_y):
            return RelationType.VERBOCEAN_MATCH

        return RelationType.NO_RELATION_FOUND

    def is_verbocean_relation(self, mention_x: MentionDataLight,
                              mention_y: MentionDataLight) -> bool:
        """
        Check if input mentions has VerbOcean relation between them

        Args:
            mention_x: MentionDataLight
            mention_y: MentionDataLight

        Returns:
            bool
        """
        x_head = mention_x.mention_head

        y_head = mention_y.mention_head

        rel = None

        if x_head in self.vo and y_head in self.vo[x_head]:
            rel = self.vo[x_head][y_head]
        elif y_head in self.vo and x_head in self.vo[y_head]:
            rel = self.vo[y_head][x_head]

        match_result = False
        if rel is not None and rel != "[unk]" and rel != "[low-vol]":
            match_result = True

        return match_result

    @staticmethod
    def get_supported_relations():
        """
        Return all supported relations by this class

        Returns:
            List[RelationType]
        """
        return [RelationType.VERBOCEAN_MATCH]

    @staticmethod
    def load_verbocean_file(fname: str) -> Dict[str, Dict[str, str]]:
        """
        Method to load referent dictionary to memory

        Returns:
            List[RelationType]
        """
        word_dict = {}
        with open(fname) as f:
            for line in f:
                word1, rel, word2, _, _ = line.strip().split()
                if word1 not in word_dict:
                    word_dict[word1] = {}
                word_dict[word1][word2] = rel
        return word_dict
