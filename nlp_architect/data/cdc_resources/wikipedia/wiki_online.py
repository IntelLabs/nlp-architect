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
import logging

from nlp_architect.data.cdc_resources.data_types.wiki.wikipedia_page import WikipediaPage
from nlp_architect.data.cdc_resources.data_types.wiki.wikipedia_page_extracted_relations import \
    WikipediaPageExtractedRelations
from nlp_architect.data.cdc_resources.wikipedia.wiki_search_page_result import \
    WikipediaSearchPageResult

os.environ['PYWIKIBOT_NO_USER_CONFIG'] = '1'

DISAMBIGUATE_PAGE = ['wikimedia disambiguation page', 'wikipedia disambiguation page']
NAME_DESCRIPTIONS = ['given name', 'first name', 'family name']

logger = logging.getLogger(__name__)


class WikiOnline(object):
    def __init__(self):
        import pywikibot
        self.pywikibot = pywikibot
        self.cache = dict()
        self.site = pywikibot.Site('en', 'wikipedia')  # The site we want to run our bot on

    def get_pages(self, phrase):
        if phrase in self.cache:
            return self.cache[phrase]

        ret_pages = set()
        word_clean = phrase.replace('-', ' ')
        word_lower = word_clean.lower()
        word_upper = word_clean.upper()
        word_title = word_clean.title()
        words_set = {phrase, word_clean, word_lower, word_upper, word_title}
        for appr in words_set:
            try:
                page_result = self.get_page_redirect(appr)
                if page_result.pageid != 0:
                    full_page = self.get_wiki_page_with_items(phrase, page_result)
                    ret_pages.add(WikipediaSearchPageResult(appr, full_page))
            except Exception as e:
                logger.error(e)

        self.cache[phrase] = ret_pages
        return ret_pages

    # pylint: disable=protected-access
    def get_wiki_page_with_items(self, phrase, page):
        item = self.get_wiki_page_item(page)
        pageid = page.pageid
        aliases = self.get_aliases(item)
        description = self.get_description(item)
        text = page.text
        page_title = page._link._title

        relations = WikipediaPageExtractedRelations()
        relations.is_disambiguation = self.is_disambiguation_page(item)
        relations.is_part_name = self.is_name_description(text, item, relations.is_disambiguation)
        relations.aliases = aliases
        relations.extract_relations_from_text_v0(text)

        ret_page = WikipediaPage(phrase, None, page_title, None, 0, pageid, description, relations)

        logger.debug('Page: {}. Extracted successfully'.format(ret_page))

        return ret_page

    def get_wiki_page_item(self, page):
        if page is not None:
            try:
                item = self.pywikibot.ItemPage.fromPage(
                    page)  # this can be used for any page object
                item.get()  # need to call it to access any data.
                return item
            except (self.pywikibot.NoPage, AttributeError, TypeError, NameError):
                pass
        return None

    def get_page_redirect(self, word):
        page = self.pywikibot.Page(self.site, word)
        if page.pageid != 0 and page.isRedirectPage():
            return page.getRedirectTarget()
        return page

    @staticmethod
    def get_aliases(item):
        if item is not None and item.aliases is not None:
            if 'en' in item.aliases:
                aliases = item.aliases['en']
                return aliases

        return None

    @staticmethod
    def get_description(item):
        description = {}
        if item is not None:
            item_desc = item.get()
            if 'desctiptions' in item_desc and 'en' in item_desc['descriptions']:
                dict([("age", 25)])
                description['descriptions'] = dict([('en', item_desc['descriptions']['en'])])

        return description

    @staticmethod
    def is_disambiguation_page(item):
        if item is not None:
            dic = item.get()
            if dic is not None and 'descriptions' in dic:
                desc = dic['descriptions']
                if desc is not None and 'en' in desc:
                    return desc['en'].lower()in DISAMBIGUATE_PAGE

        return False

    @staticmethod
    def is_name_description(text, item, is_disambiguation):
        if item is not None:
            if is_disambiguation:
                if WikipediaPageExtractedRelations.is_name_part(text):
                    return True
            else:
                dic = item.get()
                if dic is not None and 'descriptions' in dic:
                    desc = dic['descriptions']
                    if desc is not None and 'en' in desc:
                        if [s for s in NAME_DESCRIPTIONS if s in desc['en'].lower()]:
                            return True
        return False
