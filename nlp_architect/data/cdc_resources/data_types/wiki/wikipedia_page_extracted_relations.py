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
import re
import string
from typing import Set, Dict

from nlp_architect.utils.string_utils import StringUtils

PART_NAME_CATEGORIES = ['name', 'given name', 'surname']
DISAMBIGUATION_TITLE = '(disambiguation)'
DISAMBIGUATION_CATEGORY = ['disambig', 'disambiguation']


class WikipediaPageExtractedRelations(object):
    def __init__(self, is_part_name: bool = False, is_disambiguation: bool = False,
                 parenthesis: Set[str] = None,
                 disambiguation_links: Set[str] = None, categories: Set[str] = None,
                 aliases: Set[str] = None,
                 be_comp: Set[str] = None,
                 disambiguation_links_norm: Set[str] = None, categories_norm: Set[str] = None,
                 aliases_norm: Set[str] = None,
                 title_parenthesis_norm: Set[str] = None, be_comp_norm: Set[str] = None) -> None:
        """
        Object represent a Wikipedia Relations Schema

        Args:
            is_part_name (bool): Weather page title is part of a Name (ie-family name/given name..)
            is_disambiguation (bool): Weather page is a disambiguation page
            parenthesis (set): a set of all parenthesis links/titles
            disambiguation_links (set): a set of all disambiguation links/titles
            categories (set): a set of all category links/titles
            aliases (set): a set of all aliases links/titles
            be_comp (set): a set of all "is a" links/titles
            disambiguation_links_norm (set): same as disambiguation_link just normalized
            categories_norm (set): same as categories just normalized, lower and clean
            aliases_norm (set): same as aliases just normalized, lower and clean
            title_parenthesis_norm (set): same as parenthesis just normalized, lower and clean
            be_comp_norm (set): same as be_comp just normalized, lower and clean
        """
        self.is_part_name = is_part_name
        self.is_disambiguation = is_disambiguation
        self.disambiguation_links = disambiguation_links
        self.title_parenthesis = parenthesis
        self.categories = categories
        self.aliases = aliases
        self.be_comp = be_comp

        self.disambiguation_links_norm = disambiguation_links_norm
        self.categories_norm = categories_norm
        self.aliases_norm = aliases_norm
        self.title_parenthesis_norm = title_parenthesis_norm
        self.be_comp_norm = be_comp_norm

    def extract_relations_from_text_v0(self, text):
        self.disambiguation_links = set()
        self.categories = set()
        self.title_parenthesis = set()

        self.disambiguation_links_norm = set()
        self.categories_norm = set()
        self.title_parenthesis_norm = set()
        self.be_comp_norm = set()

        ext_links = set()
        title_parenthesis = set()

        text_lines = text.split('\n')
        for line in text_lines:
            cat_links = self.extract_categories(line)
            if not self.is_part_name:
                self.is_part_name = self.is_name_part(line)
                if not self.is_part_name and [s for s in PART_NAME_CATEGORIES if s in cat_links]:
                    self.is_part_name = True

            self.categories.update(cat_links)
            self.categories_norm.update(StringUtils.normalize_string_list(cat_links))

            links, parenthesis_links = self.extract_links_and_parenthesis(line)
            ext_links.update(links)
            title_parenthesis.update(parenthesis_links)

        if self.is_disambiguation:
            self.disambiguation_links = ext_links
            self.disambiguation_links_norm = StringUtils.normalize_string_list(ext_links)
            self.title_parenthesis = title_parenthesis
            self.title_parenthesis_norm = StringUtils.normalize_string_list(title_parenthesis)

    def __str__(self) -> str:
        return str(self.is_disambiguation) + ', ' + str(self.is_part_name) + ', ' + \
            str(self.disambiguation_links) + ', ' + str(self.be_comp) + ', ' + str(
            self.title_parenthesis) + ', ' + str(self.categories)

    def toJson(self) -> Dict:
        result_dict = dict()
        result_dict['isPartName'] = self.is_part_name
        result_dict['isDisambiguation'] = self.is_disambiguation

        if self.disambiguation_links is not None:
            result_dict['disambiguationLinks'] = list(self.disambiguation_links)
            result_dict['disambiguationLinksNorm'] = list(self.disambiguation_links_norm)
        if self.categories is not None:
            result_dict['categories'] = list(self.categories)
            result_dict['categoriesNorm'] = list(self.categories_norm)
        if self.aliases is not None:
            result_dict['aliases'] = list(self.aliases)
        if self.title_parenthesis is not None:
            result_dict['titleParenthesis'] = list(self.title_parenthesis)
            result_dict['titleParenthesisNorm'] = list(self.title_parenthesis_norm)
        if self.be_comp_norm is not None:
            result_dict['beCompRelations'] = list(self.be_comp)
            result_dict['beCompRelationsNorm'] = list(self.be_comp_norm)

        return result_dict

    @staticmethod
    def extract_categories(line: str) -> Set[str]:
        categories = set()
        category_form1 = re.findall(r'\[\[Category:(.*)\]\]', line)
        for cat in category_form1:
            if DISAMBIGUATION_TITLE in cat:
                cat = cat.replace(DISAMBIGUATION_TITLE, '')
            categories.add(cat)

        prog = re.search('^{{(disambig.*|Disambig.*)}}$', line)
        if prog is not None:
            category_form2 = prog.group(1)
            cats = category_form2.split('|')
            categories.update(cats)

        return categories

    @staticmethod
    def extract_links_and_parenthesis(line: str):
        links = set()
        parenthesis_links = set()
        ext_links = re.findall(r'\[\[(.*)\]\]', line)
        for link in ext_links:
            split_link = link.split('|')
            for s_link in split_link:
                parenthesis_clean = None
                matcher = re.match(r'(.*)\s?\((.*)\)', s_link)
                if matcher:
                    s_link = matcher.group(1)
                    parenthesis_match = matcher.group(2)
                    if parenthesis_match.lower() != 'disambiguation':
                        parenthesis_clean = re.sub(
                            '[' + string.punctuation + string.whitespace + ']', ' ',
                            parenthesis_match).strip()

                s_link_clean = re.sub('[' + string.punctuation + string.whitespace + ']', ' ',
                                      s_link).strip()
                if parenthesis_clean is not None and DISAMBIGUATION_TITLE not in parenthesis_clean:
                    parenthesis_links.add(parenthesis_clean)

                links.add(s_link_clean)

        return links, parenthesis_links

    @staticmethod
    def is_name_part(line: str) -> bool:
        line = line.lower()
        val = False
        if WikipediaPageExtractedRelations.find_in_line(line, '===as surname==='):
            val = True
        elif WikipediaPageExtractedRelations.find_in_line(line, '===as given name==='):
            val = True
        elif WikipediaPageExtractedRelations.find_in_line(line, '===given names==='):
            val = True
        elif WikipediaPageExtractedRelations.find_in_line(line, '==as a surname=='):
            val = True
        elif WikipediaPageExtractedRelations.find_in_line(line, '==people with the surname=='):
            val = True
        elif WikipediaPageExtractedRelations.find_in_line(line, '==family name and surname=='):
            val = True
        elif WikipediaPageExtractedRelations.find_in_line(line, 'category:given names'):
            val = True
        elif WikipediaPageExtractedRelations.find_in_line(line, '{{given name}}'):
            val = True
        return val

    @staticmethod
    def find_in_line(text: str, pattern: str) -> bool:
        found = re.findall(pattern, text)
        if found:
            return True
        return False
