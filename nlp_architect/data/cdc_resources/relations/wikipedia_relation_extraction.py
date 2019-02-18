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

from __future__ import division

import logging
import os
from typing import Set, List

from nlp_architect.common.cdc.mention_data import MentionDataLight
from nlp_architect.data.cdc_resources.data_types.wiki.wikipedia_pages import WikipediaPages
from nlp_architect.data.cdc_resources.relations.relation_extraction import RelationExtraction
from nlp_architect.data.cdc_resources.relations.relation_types_enums import RelationType, \
    WikipediaSearchMethod
from nlp_architect.data.cdc_resources.wikipedia.wiki_elastic import WikiElastic
from nlp_architect.data.cdc_resources.wikipedia.wiki_offline import WikiOffline
from nlp_architect.data.cdc_resources.wikipedia.wiki_online import WikiOnline
from nlp_architect.utils.string_utils import StringUtils

logger = logging.getLogger(__name__)


class WikipediaRelationExtraction(RelationExtraction):
    def __init__(self, method: WikipediaSearchMethod = WikipediaSearchMethod.ONLINE,
                 wiki_file: str = None, host: str = None, port: int = None,
                 index: str = None, filter_pronouns: bool = True,
                 filter_time_data: bool = True) -> None:
        """
        Extract Relation between two mentions according to Wikipedia knowledge

        Args:
            method (optional): WikipediaSearchMethod.{ONLINE/OFFLINE/ELASTIC} run against wiki
                site a sub-set of wiki or on a local elastic database (default = ONLINE)
            wiki_file (required on OFFLINE mode): str Location of Wikipedia file to work with
            host (required on Elastic mode): str the Elastic search host name
            port (required on Elastic mode): int the Elastic search port number
            index (required on Elastic mode): int the Elastic search index name
        """
        logger.info('Loading Wikipedia module')
        self.filter_pronouns = filter_pronouns
        self.filter_time_data = filter_time_data
        connectivity = method
        if connectivity == WikipediaSearchMethod.ONLINE:
            self.pywiki_impl = WikiOnline()
        elif connectivity == WikipediaSearchMethod.OFFLINE:
            if wiki_file is not None and os.path.isdir(wiki_file):
                self.pywiki_impl = WikiOffline(wiki_file)
            else:
                raise FileNotFoundError('Wikipedia resource file not found or not in path, '
                                        'create it or change the initialization method')
        elif connectivity == WikipediaSearchMethod.ELASTIC:
            self.pywiki_impl = WikiElastic(host, port, index)

        logger.info('Wikipedia module lead successfully')
        super(WikipediaRelationExtraction, self).__init__()

    def get_phrase_related_pages(self, mention_str: str) -> WikipediaPages:
        """
        Get all WikipediaPages pages related with this mention string

        Args:
            mention_str: str

        Returns:
            WikipediaPages
        """
        pages = self.pywiki_impl.get_pages(mention_str.strip())
        ret_pages = WikipediaPages()
        if pages:
            for search_page in pages:
                if search_page.page_result.pageid != 0:
                    ret_pages.add_page(search_page.page_result)

        return ret_pages

    def extract_all_relations(self, mention_x: MentionDataLight,
                              mention_y: MentionDataLight) -> Set[RelationType]:
        """
        Try to find if mentions has anyone or more of the relations this class support

        Args:
            mention_x: MentionDataLight
            mention_y: MentionDataLight

        Returns:
            Set[RelationType]: One or more of: RelationType.WIKIPEDIA_BE_COMP,
                RelationType.WIKIPEDIA_TITLE_PARENTHESIS,
                RelationType.WIKIPEDIA_DISAMBIGUATION, RelationType.WIKIPEDIA_CATEGORY,
                RelationType.WIKIPEDIA_REDIRECT_LINK, RelationType.WIKIPEDIA_ALIASES,
                RelationType.WIKIPEDIA_PART_OF_SAME_NAME
        """
        relations = set()
        mention1_str = mention_x.tokens_str.strip()
        mention2_str = mention_y.tokens_str.strip()

        if self.filter_pronouns:
            if self.is_both_opposite_personal_pronouns(mention1_str, mention2_str):
                relations.add(RelationType.NO_RELATION_FOUND)
                return relations

        if self.filter_time_data:
            if self.is_both_data_or_time(mention_x, mention_y):
                relations.add(RelationType.NO_RELATION_FOUND)
                return relations

        pages1 = self.get_phrase_related_pages(mention1_str)
        pages2 = self.get_phrase_related_pages(mention2_str)

        # check if search phrase is empty meaning it is probably a stop word
        if pages1.is_empty_norm_phrase or pages2.is_empty_norm_phrase:
            relations.add(RelationType.NO_RELATION_FOUND)
            return relations

        if self.is_redirect_same(pages1, pages2):
            relations.add(RelationType.WIKIPEDIA_REDIRECT_LINK)

        titles1 = pages1.get_and_set_titles()
        titles1.add(mention1_str + ' ' + mention2_str)
        titles1.add(mention2_str + ' ' + mention1_str)

        titles2 = pages2.get_and_set_titles()
        titles2.add(mention1_str + ' ' + mention2_str)
        titles2.add(mention2_str + ' ' + mention1_str)

        relation_alias = self.extract_aliases(pages1, pages2, titles1, titles2)
        if relation_alias is not RelationType.NO_RELATION_FOUND:
            relations.add(relation_alias)
        relation_dis = self.extract_disambig(pages1, pages2, titles1, titles2)
        if relation_dis is not RelationType.NO_RELATION_FOUND:
            relations.add(relation_dis)
        relation_cat = self.extract_category(pages1, pages2, titles1, titles2)
        if relation_cat is not RelationType.NO_RELATION_FOUND:
            relations.add(relation_cat)
        relation_par = self.extract_parenthesis(pages1, pages2, titles1, titles2)
        if relation_par is not RelationType.NO_RELATION_FOUND:
            relations.add(relation_par)
        relation_be = self.extract_be_comp(pages1, pages2, titles1, titles2)
        if relation_be is not RelationType.NO_RELATION_FOUND:
            relations.add(relation_be)

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
        mention1_str = mention_x.tokens_str.strip()
        mention2_str = mention_y.tokens_str.strip()

        if self.filter_pronouns:
            if self.is_both_opposite_personal_pronouns(mention1_str, mention2_str):
                return RelationType.NO_RELATION_FOUND

        if self.filter_time_data:
            if self.is_both_data_or_time(mention_x, mention_y):
                return RelationType.NO_RELATION_FOUND

        pages1 = self.get_phrase_related_pages(mention1_str)
        pages2 = self.get_phrase_related_pages(mention2_str)

        # check if search phrase is empty meaning it is probably a stop word
        if pages1.is_empty_norm_phrase or pages2.is_empty_norm_phrase:
            return RelationType.NO_RELATION_FOUND

        if relation == RelationType.WIKIPEDIA_REDIRECT_LINK:
            if self.is_redirect_same(pages1, pages2):
                return RelationType.WIKIPEDIA_REDIRECT_LINK

            return RelationType.NO_RELATION_FOUND

        titles1 = pages1.get_and_set_titles()
        titles1.add(mention1_str + ' ' + mention2_str)
        titles1.add(mention2_str + ' ' + mention1_str)

        titles2 = pages2.get_and_set_titles()
        titles2.add(mention1_str + ' ' + mention2_str)
        titles2.add(mention2_str + ' ' + mention1_str)

        if relation == RelationType.WIKIPEDIA_ALIASES:
            return self.extract_aliases(pages1, pages2, titles1, titles2)
        if relation == RelationType.WIKIPEDIA_DISAMBIGUATION:
            return self.extract_disambig(pages1, pages2, titles1, titles2)
        if relation == RelationType.WIKIPEDIA_CATEGORY:
            return self.extract_category(pages1, pages2, titles1, titles2)
        if relation == RelationType.WIKIPEDIA_TITLE_PARENTHESIS:
            return self.extract_parenthesis(pages1, pages2, titles1, titles2)
        if relation == RelationType.WIKIPEDIA_BE_COMP:
            return self.extract_be_comp(pages1, pages2, titles1, titles2)

        return RelationType.NO_RELATION_FOUND

    @staticmethod
    def extract_be_comp(pages1: WikipediaPages, pages2: WikipediaPages, titles1: Set[str],
                        titles2: Set[str]) -> RelationType:
        """
        Check if input mentions has be-comp/is-a relation

        Args:
            pages1: WikipediaPages
            pages2: WikipediaPage
            titles1: Set[str]
            titles2: Set[str]

        Returns:
            RelationType.WIKIPEDIA_BE_COMP or RelationType.NO_RELATION_FOUND
        """
        relation = RelationType.NO_RELATION_FOUND
        if bool(pages1.get_and_set_be_comp() & titles2):
            relation = RelationType.WIKIPEDIA_BE_COMP
        elif bool(pages2.get_and_set_be_comp() & titles1):
            relation = RelationType.WIKIPEDIA_BE_COMP
        return relation

    @staticmethod
    def extract_parenthesis(pages1: WikipediaPages, pages2: WikipediaPages, titles1: Set[str],
                            titles2: Set[str]) -> RelationType:
        """
        Check if input mentions has parenthesis relation

        Args:
            pages1: WikipediaPages
            pages2: WikipediaPage
            titles1: Set[str]
            titles2: Set[str]

        Returns:
            RelationType.WIKIPEDIA_TITLE_PARENTHESIS or RelationType.NO_RELATION_FOUND
        """
        relation = RelationType.NO_RELATION_FOUND
        if bool(pages1.get_and_set_parenthesis() & titles2):
            relation = RelationType.WIKIPEDIA_TITLE_PARENTHESIS
        elif bool(pages2.get_and_set_parenthesis() & titles1):
            relation = RelationType.WIKIPEDIA_TITLE_PARENTHESIS
        return relation

    @staticmethod
    def extract_category(pages1: WikipediaPages, pages2: WikipediaPages, titles1: Set[str],
                         titles2: Set[str]) -> RelationType:
        """
        Check if input mentions has category relation

        Args:
            pages1: WikipediaPages
            pages2: WikipediaPage
            titles1: Set[str]
            titles2: Set[str]

        Returns:
            RelationType.WIKIPEDIA_CATEGORY or RelationType.NO_RELATION_FOUND
        """
        relation = RelationType.NO_RELATION_FOUND
        if bool(pages1.get_and_set_all_categories() & titles2):
            relation = RelationType.WIKIPEDIA_CATEGORY
        elif bool(pages2.get_and_set_all_categories() & titles1):
            relation = RelationType.WIKIPEDIA_CATEGORY
        return relation

    @staticmethod
    def extract_disambig(pages1: WikipediaPages, pages2: WikipediaPages, titles1: Set[str],
                         titles2: Set[str]) -> RelationType:
        """
        Check if input mentions has disambiguation relation

        Args:
            pages1: WikipediaPages
            pages2: WikipediaPage
            titles1: Set[str]
            titles2: Set[str]

        Returns:
            RelationType.WIKIPEDIA_DISAMBIGUATION or RelationType.NO_RELATION_FOUND
        """
        relation = RelationType.NO_RELATION_FOUND
        if bool(pages1.get_and_set_all_disambiguation() & titles2):
            relation = RelationType.WIKIPEDIA_DISAMBIGUATION
        elif bool(pages2.get_and_set_all_disambiguation() & titles1):
            relation = RelationType.WIKIPEDIA_DISAMBIGUATION
        return relation

    @staticmethod
    def extract_aliases(pages1: WikipediaPages, pages2: WikipediaPages, titles1: Set[str],
                        titles2: Set[str]) -> RelationType:
        """
        Check if input mentions has aliases relation

        Args:
            pages1: WikipediaPages
            pages2: WikipediaPage
            titles1: Set[str]
            titles2: Set[str]
        Returns:
            RelationType.WIKIPEDIA_ALIASES or RelationType.NO_RELATION_FOUND
        """
        relation = RelationType.NO_RELATION_FOUND
        if bool(pages1.get_and_set_all_aliases() & titles2):
            relation = RelationType.WIKIPEDIA_ALIASES
        elif bool(pages2.get_and_set_all_aliases() & titles1):
            relation = RelationType.WIKIPEDIA_ALIASES
        return relation

    def is_part_of_same_name(self, pages1: WikipediaPages, pages2: WikipediaPages) -> bool:
        """
        Check if input mentions has part of same name relation (eg: page1=John, page2=Smith)

        Args:
            pages1: WikipediaPages
            pages2: WikipediaPage

        Returns:
            bool
        """
        for page1 in pages1.pages:
            for page2 in pages2.pages:
                if page1.relations.is_part_name and page2.relations.is_part_name:
                    pages = self.pywiki_impl.get_pages(page1.orig_phrase + ' ' + page2.orig_phrase)
                    for page in pages:
                        if page.page_result.pageid != 0:
                            return True
        return False

    @staticmethod
    def is_redirect_same(pages1: WikipediaPages, pages2: WikipediaPages) -> bool:
        """
        Check if input mentions has same wikipedia redirect page

        Args:
            pages1: WikipediaPages
            pages2: WikipediaPage

        Returns:
            bool
        """
        for page1 in pages1.get_pages():
            for page2 in pages2.get_pages():
                if page1.pageid > 0 and page2.pageid > 0:
                    if page1.pageid == page2.pageid:
                        return True
        return False

    @staticmethod
    def get_supported_relations() -> List[RelationType]:
        """
        Return all supported relations by this class

        Returns:
            List[RelationType]
        """
        return [RelationType.WIKIPEDIA_BE_COMP, RelationType.WIKIPEDIA_TITLE_PARENTHESIS,
                RelationType.WIKIPEDIA_DISAMBIGUATION, RelationType.WIKIPEDIA_CATEGORY,
                RelationType.WIKIPEDIA_REDIRECT_LINK, RelationType.WIKIPEDIA_ALIASES,
                RelationType.WIKIPEDIA_PART_OF_SAME_NAME]

    @staticmethod
    def is_both_opposite_personal_pronouns(phrase1: str, phrase2: str) -> bool:
        """
        check if both phrases refers to pronouns

        Returns:
            bool
        """
        result = False
        if StringUtils.is_pronoun(phrase1.lower()) and StringUtils.is_pronoun(phrase2.lower()):
            result = True

        return result

    @staticmethod
    def is_both_data_or_time(mention1: MentionDataLight, mention2: MentionDataLight) -> bool:
        """
        check if both phrases refers to time or date

        Returns:
            bool
        """
        mention1_ner = mention1.mention_ner
        mention2_ner = mention2.mention_ner

        if mention1_ner is None:
            _, _, _, mention1_ner = StringUtils.find_head_lemma_pos_ner(mention1.tokens_str)
        if mention2_ner is None:
            _, _, _, mention2_ner = StringUtils.find_head_lemma_pos_ner(mention2.tokens_str)

        is1_time_or_data = 'DATE' in mention1_ner or 'TIME' in mention1_ner
        is2_time_or_data = 'DATE' in mention2_ner or 'TIME' in mention2_ner

        result = False
        if is1_time_or_data and is2_time_or_data:
            result = True

        return result
