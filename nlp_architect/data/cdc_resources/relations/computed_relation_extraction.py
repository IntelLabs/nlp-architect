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
import difflib
import logging
from typing import Set, List

from num2words import num2words

from nlp_architect.common.cdc.mention_data import MentionDataLight
from nlp_architect.data.cdc_resources.relations.relation_extraction import RelationExtraction
from nlp_architect.data.cdc_resources.relations.relation_types_enums import RelationType
from nlp_architect.utils.string_utils import StringUtils

logger = logging.getLogger(__name__)


class ComputedRelationExtraction(RelationExtraction):
    """
    Extract Relation between two mentions according to computation and rule based algorithms
    """
    def extract_all_relations(self, mention_x: MentionDataLight,
                              mention_y: MentionDataLight) -> Set[RelationType]:
        """
        Try to find if mentions has anyone or more of the relations this class support

        Args:
            mention_x: MentionDataLight
            mention_y: MentionDataLight

        Returns:
            Set[RelationType]: One or more of: RelationType.EXACT_STRING, RelationType.FUZZY_FIT,
                RelationType.FUZZY_HEAD_FIT, RelationType.SAME_HEAD_LEMMA,
                RelationType.SAME_HEAD_LEMMA_RELAX
        """
        relations = set()
        mention_x_str = mention_x.tokens_str
        mention_y_str = mention_y.tokens_str

        if StringUtils.is_pronoun(mention_x_str.lower()) or StringUtils.is_pronoun(
                mention_y_str.lower()):
            relations.add(RelationType.NO_RELATION_FOUND)
            return relations

        exact_rel = self.extract_exact_string(mention_x, mention_y)
        fuzzy_rel = self.extract_fuzzy_fit(mention_x, mention_y)
        fuzzy_head_rel = self.extract_fuzzy_head_fit(mention_x, mention_y)
        same_head_rel = self.extract_same_head_lemma(mention_x, mention_y)
        if exact_rel != RelationType.NO_RELATION_FOUND:
            relations.add(exact_rel)
        if fuzzy_rel != RelationType.NO_RELATION_FOUND:
            relations.add(fuzzy_rel)
        if fuzzy_head_rel != RelationType.NO_RELATION_FOUND:
            relations.add(fuzzy_head_rel)
        if same_head_rel != RelationType.NO_RELATION_FOUND:
            relations.add(same_head_rel)

        if len(relations) == 0:
            relations.add(RelationType.NO_RELATION_FOUND)

        return relations

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
        mention_x_str = mention_x.tokens_str
        mention_y_str = mention_y.tokens_str

        if StringUtils.is_pronoun(mention_x_str.lower()) or StringUtils.is_pronoun(
                mention_y_str.lower()):
            return RelationType.NO_RELATION_FOUND

        if relation == RelationType.EXACT_STRING:
            return self.extract_exact_string(mention_x, mention_y)
        if relation == RelationType.FUZZY_FIT:
            return self.extract_fuzzy_fit(mention_x, mention_y)
        if relation == RelationType.FUZZY_HEAD_FIT:
            return self.extract_fuzzy_head_fit(mention_x, mention_y)
        if relation == RelationType.SAME_HEAD_LEMMA:
            is_same_lemma = self.extract_same_head_lemma(mention_x, mention_y)
            if is_same_lemma != RelationType.NO_RELATION_FOUND:
                return relation

        return RelationType.NO_RELATION_FOUND

    @staticmethod
    def extract_same_head_lemma(mention_x: MentionDataLight,
                                mention_y: MentionDataLight) -> RelationType:
        """
        Check if input mentions has same head lemma relation

        Args:
            mention_x: MentionDataLight
            mention_y: MentionDataLight

        Returns:
            RelationType.SAME_HEAD_LEMMA or RelationType.NO_RELATION_FOUND
        """
        if StringUtils.is_preposition(mention_x.mention_head_lemma) or \
                StringUtils.is_preposition(mention_y.mention_head_lemma) or \
                StringUtils.is_determiner(mention_x.mention_head_lemma) or \
                StringUtils.is_determiner(mention_y.mention_head_lemma):
            return RelationType.NO_RELATION_FOUND
        if mention_x.mention_head_lemma == mention_y.mention_head_lemma:
            return RelationType.SAME_HEAD_LEMMA
        return RelationType.NO_RELATION_FOUND

    @staticmethod
    def extract_fuzzy_head_fit(mention_x: MentionDataLight,
                               mention_y: MentionDataLight) -> RelationType:
        """
        Check if input mentions has fuzzy head fit relation

        Args:
            mention_x: MentionDataLight
            mention_y: MentionDataLight

        Returns:
            RelationType.FUZZY_HEAD_FIT or RelationType.NO_RELATION_FOUND
        """
        if StringUtils.is_preposition(mention_x.mention_head_lemma.lower()) or \
                StringUtils.is_preposition(mention_y.mention_head_lemma.lower()):
            return RelationType.NO_RELATION_FOUND

        mention_y_tokens = mention_y.tokens_str.split()
        mention_x_tokens = mention_x.tokens_str.split()
        if mention_x.mention_head in mention_y_tokens or \
                mention_y.mention_head in mention_x_tokens:
            return RelationType.FUZZY_HEAD_FIT
        return RelationType.NO_RELATION_FOUND

    @staticmethod
    def extract_fuzzy_fit(mention_x: MentionDataLight,
                          mention_y: MentionDataLight) -> RelationType:
        """
        Check if input mentions has fuzzy fit relation

        Args:
            mention_x: MentionDataLight
            mention_y: MentionDataLight

        Returns:
            RelationType.FUZZY_FIT or RelationType.NO_RELATION_FOUND
        """
        relation = RelationType.NO_RELATION_FOUND
        mention1_str = mention_x.tokens_str
        mention2_str = mention_y.tokens_str
        if difflib.SequenceMatcher(None, mention1_str, mention2_str).ratio() * 100 >= 90:
            relation = RelationType.FUZZY_FIT
            return relation

        # Convert numbers to words
        x_words = [num2words(int(w)).replace('-', ' ') if w.isdigit() else w for w in
                   mention1_str.split()]
        y_words = [num2words(int(w)).replace('-', ' ') if w.isdigit() else w for w in
                   mention2_str.split()]

        fuzzy_result = difflib.SequenceMatcher(None, ' '.join(x_words),
                                               ' '.join(y_words)).ratio() * 100 >= 85
        if fuzzy_result:
            relation = RelationType.FUZZY_FIT
        return relation

    @staticmethod
    def extract_exact_string(mention_x: MentionDataLight,
                             mention_y: MentionDataLight) -> RelationType:
        """
        Check if input mentions has exact string relation

        Args:
            mention_x: MentionDataLight
            mention_y: MentionDataLight

        Returns:
            RelationType.EXACT_STRING or RelationType.NO_RELATION_FOUND
        """
        relation = RelationType.NO_RELATION_FOUND
        mention1_str = mention_x.tokens_str
        mention2_str = mention_y.tokens_str
        if StringUtils.is_preposition(mention1_str.lower()) or \
                StringUtils.is_preposition(mention2_str.lower()):
            return relation

        if mention1_str.lower() == mention2_str.lower():
            relation = RelationType.EXACT_STRING

        return relation

    @staticmethod
    def get_supported_relations() -> List[RelationType]:
        """
        Return all supported relations by this class

        Returns:
            List[RelationType]
        """
        return [RelationType.EXACT_STRING, RelationType.FUZZY_FIT, RelationType.FUZZY_HEAD_FIT,
                RelationType.SAME_HEAD_LEMMA]
