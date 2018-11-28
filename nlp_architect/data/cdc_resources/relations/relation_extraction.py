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
from typing import List

from nlp_architect.common.cdc.mention_data import MentionDataLight
from nlp_architect.data.cdc_resources.relations.relation_types_enums import RelationType


class RelationExtraction(object):
    def __init__(self):
        pass

    def extract_relation(self, mention_x: MentionDataLight, mention_y: MentionDataLight,
                         relation: RelationType) -> RelationType:
        """
        Base Class Check if Sub class support given relation before executing the sub class

        Args:
            mention_x: MentionDataLight
            mention_y: MentionDataLight
            relation: RelationType

        Returns:
            RelationType: relation in case mentions has given relation and
                RelationType.NO_RELATION_FOUND otherwise
        """
        ret_relation = RelationType.NO_RELATION_FOUND
        if relation in self.get_supported_relations():
            ret_relation = self.extract_sub_relations(mention_x, mention_y, relation)
        return ret_relation

    def extract_sub_relations(self, mention_x: MentionDataLight, mention_y: MentionDataLight,
                              relation: RelationType) -> RelationType:
        raise NotImplementedError

    @staticmethod
    def get_supported_relations() -> List[RelationType]:
        raise NotImplementedError
