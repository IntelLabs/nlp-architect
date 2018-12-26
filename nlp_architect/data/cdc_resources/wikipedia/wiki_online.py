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
import re

from nlp_architect.data.cdc_resources.data_types.wiki.wikipedia_page import WikipediaPage
from nlp_architect.data.cdc_resources.data_types.wiki.wikipedia_page_extracted_relations import \
    WikipediaPageExtractedRelations
from nlp_architect.data.cdc_resources.wikipedia.wiki_search_page_result import \
    WikipediaSearchPageResult
from nlp_architect.utils.text import SpacyInstance

os.environ['PYWIKIBOT_NO_USER_CONFIG'] = '1'

DISAMBIGUATE_PAGE = ['wikimedia disambiguation page', 'wikipedia disambiguation page']
NAME_DESCRIPTIONS = ['given name', 'first name', 'family name']

logger = logging.getLogger(__name__)


class WikiOnline(object):
    def __init__(self):
        import pywikibot
        self.spacy = SpacyInstance()
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
        relations.be_comp, relations.be_comp_norm = self.extract_be_comp(text)
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

    # pylint: disable=no-else-return
    def extract_be_comp(self, text):
        first_sentence_start_index = text.index("'''")
        if first_sentence_start_index >= 0:
            last_temp_index = text.find('\n', first_sentence_start_index)
        if last_temp_index == -1:
            last_temp_index = len(text)

        first_paragraph = text[first_sentence_start_index:last_temp_index]
        if WikiOnline.extract_be_a_index(first_paragraph) == -1 and last_temp_index != len(text):
            return self.extract_be_comp(text[last_temp_index:])
        elif last_temp_index == len(text):
            return None, None

        first_paragraph_clean = re.sub(r'\([^)]*\)', '', first_paragraph)
        first_paragraph_clean = re.sub(r'<[^>]*>', '', first_paragraph_clean)
        first_paragraph_clean = re.sub(r'{[^}]*}', '', first_paragraph_clean)
        first_paragraph_clean = re.sub(r'\[\[[^]]*\]\]', '', first_paragraph_clean)
        first_paragraph_clean = re.sub(r'[\']', '', first_paragraph_clean)
        first_paragraph_clean = re.sub(r'&nbsp;', ' ', first_paragraph_clean)

        return self.extract_be_comp_relations(first_paragraph_clean)

    # pylint: disable=not-callable
    def extract_be_comp_relations(self, first_paragraph):
        be_comp = set()
        be_comp_norm = set()
        if first_paragraph:
            doc = self.spacy.parser(first_paragraph)
            for token in doc:
                target = token.text
                target_lemma = token.lemma_
                relation = token.dep_
                governor = token.head.text
                governor_lemma = token.head.lemma_
                if relation == 'acl':
                    break
                if relation == 'punct' and target == '.':
                    break
                elif relation == 'cop':
                    be_comp.add(governor)
                    be_comp_norm.add(governor_lemma)
                elif relation == 'nsubj':
                    be_comp.add(target)
                    be_comp_norm.add(target_lemma)
                elif relation == 'dep':
                    be_comp.add(governor)
                    be_comp_norm.add(governor_lemma)
                elif relation == 'compound':
                    be_comp.add(target + ' ' + governor)
                    be_comp_norm.add(target_lemma + ' ' + governor_lemma)
                elif relation == 'amod':
                    be_comp.add(target + ' ' + governor)
                    be_comp_norm.add(target_lemma + ' ' + governor_lemma)
                elif relation in ['conj', 'appos']:
                    be_comp.add(target)
                    be_comp_norm.add(target_lemma)

        return be_comp, be_comp_norm

    @staticmethod
    def extract_be_a_index(sentence):
        result = None
        if 'is a' in sentence:
            result = sentence.index("is a")
        elif 'are a' in sentence:
            result = sentence.index("are a")
        elif 'was a' in sentence:
            result = sentence.index("was a")
        elif 'were a' in sentence:
            result = sentence.index("were a")
        elif 'be a' in sentence:
            result = sentence.index("be a")
        elif 'is the' in sentence:
            result = sentence.index("is the")
        elif 'are the' in sentence:
            result = sentence.index("are the")
        elif 'was the' in sentence:
            result = sentence.index("was the")
        elif 'were the' in sentence:
            result = sentence.index("were the")
        elif 'be the' in sentence:
            result = sentence.index("be the")

        return result
