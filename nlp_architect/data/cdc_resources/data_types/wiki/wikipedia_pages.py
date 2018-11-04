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


class WikipediaPages(object):
    def __init__(self):
        """
        Object represent a set of Wikipedia Pages
        """
        self.pages = set()
        self.is_empty_norm_phrase = True

    def get_pages(self):
        return self.pages

    def add_page(self, page):
        self.pages.add(page)
        if page.orig_phrase_norm is not None and page.orig_phrase_norm != '':
            self.is_empty_norm_phrase = False

    def get_and_set_all_disambiguation(self):
        all_disambiguations = []
        for page in self.pages:
            if page.relations.disambiguation_links_norm is not None:
                all_disambiguations.extend(page.relations.disambiguation_links_norm)
            if page.relations.disambiguation_links is not None:
                all_disambiguations.extend(page.relations.disambiguation_links)
        return set(all_disambiguations)

    def get_and_set_all_categories(self):
        all_categories = []
        for page in self.pages:
            if page.relations.categories_norm is not None:
                all_categories.extend(page.relations.categories_norm)
            if page.relations.categories is not None:
                all_categories.extend(page.relations.categories)
        return set(all_categories)

    def get_and_set_all_aliases(self):
        all_aliases = []
        for page in self.pages:
            if page.relations.aliases_norm is not None:
                all_aliases.extend(page.relations.aliases_norm)
            if page.relations.aliases is not None:
                all_aliases.extend(page.relations.aliases)
        return set(all_aliases)

    def get_and_set_parenthesis(self):
        all_parenthesis = []
        for page in self.pages:
            if page.relations.title_parenthesis_norm is not None:
                all_parenthesis.extend(page.relations.title_parenthesis_norm)
            if page.relations.title_parenthesis is not None:
                all_parenthesis.extend(page.relations.title_parenthesis)
        return set(all_parenthesis)

    def get_and_set_be_comp(self):
        all_be_comp = []
        for page in self.pages:
            if page.relations.be_comp_norm is not None:
                all_be_comp.extend(page.relations.be_comp_norm)
            if page.relations.be_comp is not None:
                all_be_comp.extend(page.relations.be_comp)
        return set(all_be_comp)

    def get_and_set_titles(self):
        all_titles = []
        for page in self.pages:
            if page.orig_phrase != '':
                all_titles.append(page.orig_phrase)
                all_titles.append(page.orig_phrase_norm)
            if page.wiki_title != '':
                all_titles.append(page.wiki_title)
                all_titles.append(page.wiki_title_norm)
        return set(all_titles)

    def toJson(self):
        result_dict = {}
        page_list = []
        for page in self.pages:
            page_list.append(page.toJson())

        result_dict['pages'] = page_list
        return result_dict

    def __str__(self) -> str:
        result_str = ''
        for page in self.pages:
            result_str += str(page) + ', '

        return result_str.strip()
