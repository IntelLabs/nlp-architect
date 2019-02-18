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
from typing import Set, List

from nlp_architect.common.cdc.mention_data import MentionDataLight
from nlp_architect.data.cdc_resources.data_types.wn.wordnet_page import WordnetPage
from nlp_architect.data.cdc_resources.relations.relation_extraction import RelationExtraction
from nlp_architect.data.cdc_resources.relations.relation_types_enums import RelationType, \
    OnlineOROfflineMethod
from nlp_architect.data.cdc_resources.wordnet.wordnet_offline import WordnetOffline
from nlp_architect.data.cdc_resources.wordnet.wordnet_online import WordnetOnline
from nlp_architect.utils.string_utils import StringUtils

logger = logging.getLogger(__name__)


class WordnetRelationExtraction(RelationExtraction):
    def __init__(self, method: OnlineOROfflineMethod = OnlineOROfflineMethod.ONLINE,
                 wn_file: str = None):
        """
        Extract Relation between two mentions according to Word Embedding cosine distance

        Args:
            method (required): OnlineOROfflineMethod.{ONLINE/OFFLINE} run against full wordnet or
                a sub-set of it (default = ONLINE)
            wn_file (required on OFFLINE mode): str Location of wordnet subset file to work with
        """
        logger.info('Loading Wordnet module')
        self.connectivity = method
        if self.connectivity == OnlineOROfflineMethod.ONLINE:
            self.wordnet_impl = WordnetOnline()
        elif self.connectivity == OnlineOROfflineMethod.OFFLINE:
            if wn_file is not None and os.path.isdir(wn_file):
                self.wordnet_impl = WordnetOffline(wn_file)
            else:
                raise FileNotFoundError('WordNet resource directory not found or not in path')

        logger.info('Wordnet module lead successfully')
        super(WordnetRelationExtraction, self).__init__()

    def extract_all_relations(self, mention_x: MentionDataLight,
                              mention_y: MentionDataLight) -> Set[RelationType]:
        """
        Try to find if mentions has anyone or more of the relations this class support

        Args:
            mention_x: MentionDataLight
            mention_y: MentionDataLight

        Returns:
            Set[RelationType]: One or more of: RelationType.WORDNET_SAME_SYNSET_ENTITY,
                RelationType.WORDNET_SAME_SYNSET_EVENT, RelationType.WORDNET_PARTIAL_SYNSET_MATCH,
                RelationType.WORDNET_DERIVATIONALLY
        """
        relations = set()
        mention_x_str = mention_x.tokens_str
        mention_y_str = mention_y.tokens_str
        if StringUtils.is_pronoun(mention_x_str.lower()) or StringUtils.is_pronoun(
                mention_y_str.lower()):
            relations.add(RelationType.NO_RELATION_FOUND)
            return relations

        page_x = self.wordnet_impl.get_pages(mention_x)
        page_y = self.wordnet_impl.get_pages(mention_y)

        if page_x and page_y:
            deriv_rel = self.extract_derivation(page_x, page_y)
            part_syn_rel = self.extract_partial_synset_match(page_x, page_y)
            same_syn_rel = self.extract_same_synset_entity(page_x, page_y)
            if deriv_rel != RelationType.NO_RELATION_FOUND:
                relations.add(deriv_rel)
            if part_syn_rel != RelationType.NO_RELATION_FOUND:
                relations.add(part_syn_rel)
            if same_syn_rel != RelationType.NO_RELATION_FOUND:
                relations.add(same_syn_rel)

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

        page_x = self.wordnet_impl.get_pages(mention_x)
        page_y = self.wordnet_impl.get_pages(mention_y)

        if page_x and page_y:
            if relation == RelationType.WORDNET_DERIVATIONALLY:
                return self.extract_derivation(page_x, page_y)
            if relation == RelationType.WORDNET_PARTIAL_SYNSET_MATCH:
                return self.extract_partial_synset_match(page_x, page_y)
            if relation == RelationType.WORDNET_SAME_SYNSET:
                return self.extract_same_synset_entity(page_x, page_y)

        return RelationType.NO_RELATION_FOUND

    @staticmethod
    def extract_derivation(page_x: WordnetPage, page_y: WordnetPage) -> RelationType:
        """
        Check if input mentions has derivation relation

        Args:
            page_x:WordnetPage
            page_y:WordnetPage

        Returns:
            RelationType.WORDNET_DERIVATIONALLY or RelationType.NO_RELATION_FOUND
        """
        x_head = page_x.head
        x_head_lemma = page_x.head_lemma
        y_head = page_y.head
        y_head_lemma = page_y.head_lemma

        x_set = set()
        x_set.update(page_x.head_derivationally)
        x_set.update(page_x.head_lemma_derivationally)

        y_set = set()
        y_set.update(page_y.head_derivationally)
        y_set.update(page_y.head_lemma_derivationally)

        relation = RelationType.NO_RELATION_FOUND

        if y_head in x_set or y_head_lemma in x_set or x_head in y_set or \
                x_head_lemma in y_set or len(x_set & y_set) > 0:
            relation = RelationType.WORDNET_DERIVATIONALLY
            # print 'matched by derivation - ' + str(x_head)+ ' , ' + str(y_head)

        return relation

    @staticmethod
    def extract_partial_synset_match(page_x: WordnetPage, page_y: WordnetPage) -> RelationType:
        """
        Check if input mentions has partial synset relation

        Args:
            page_x:WordnetPage
            page_y:WordnetPage

        Returns:
            RelationType.WORDNET_PARTIAL_SYNSET_MATCH or RelationType.NO_RELATION_FOUND
        """
        x_words = page_x.clean_phrase.split()
        y_words = page_y.clean_phrase.split()

        if len(x_words) == 0 or len(y_words) == 0:
            return RelationType.NO_RELATION_FOUND

        x_synonyms = page_x.all_clean_words_synonyms
        y_synonyms = page_y.all_clean_words_synonyms

        # One word - check whether there is intersection between synsets
        if len(x_synonyms) == 1 and len(y_synonyms) == 1 and \
                len([w for w in (x_synonyms[0] & y_synonyms[0])]) > 0:
            # print 'matched by partial - ' + str(y) + ' , ' + str(x)
            return RelationType.WORDNET_PARTIAL_SYNSET_MATCH

        return RelationType.NO_RELATION_FOUND

    @staticmethod
    def extract_same_synset_entity(page_x: WordnetPage, page_y: WordnetPage) -> RelationType:
        """
        Check if input mentions has same synset relation for entity mentions

        Args:
            page_x:WordnetPage
            page_y:WordnetPage

        Returns:
            RelationType.WORDNET_SAME_SYNSET_ENTITY or RelationType.NO_RELATION_FOUND
        """
        match_result = RelationType.NO_RELATION_FOUND
        th = 0
        if len([w for w in (page_x.head_synonyms & page_y.head_synonyms)]) > th:
            match_result = RelationType.WORDNET_SAME_SYNSET

        return match_result

    @staticmethod
    def get_supported_relations() -> List[RelationType]:
        """
        Return all supported relations by this class

        Returns:
            List[RelationType]
        """
        return [RelationType.WORDNET_SAME_SYNSET,
                RelationType.WORDNET_PARTIAL_SYNSET_MATCH, RelationType.WORDNET_DERIVATIONALLY]
