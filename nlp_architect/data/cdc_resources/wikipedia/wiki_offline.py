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
from os import listdir
from os.path import join, isfile

from nlp_architect.data.cdc_resources.data_types.wiki.wikipedia_page import WikipediaPage
from nlp_architect.data.cdc_resources.data_types.wiki.wikipedia_page_extracted_relations import \
    WikipediaPageExtractedRelations
from nlp_architect.data.cdc_resources.wikipedia.wiki_search_page_result import \
    WikipediaSearchPageResult
from nlp_architect.utils.io import load_json_file

logger = logging.getLogger(__name__)


class WikiOffline(object):
    def __init__(self, wikidump):
        if wikidump:
            self.dump = self.load_dump(wikidump)
            logger.info('Wikipedia dump loaded successfully!')

    def get_pages(self, phrase):
        if phrase and phrase in self.dump:
            pages = self.dump[phrase]
            if pages:
                return pages

        return set()

    @staticmethod
    def extract_json_values(json_pages):
        pages = set()
        for json_page in json_pages:
            description = json_page.get('description', None)
            pageid = int(json_page.get('pageid', 0))
            orig_phrase = json_page.get('orig_phrase', None)
            orig_phrase_norm = json_page.get('orig_phrase_norm', None)
            wiki_title = json_page.get('wiki_title', None)
            wiki_title_norm = json_page.get('wiki_title_norm', None)

            relations_json = json_page.get('relations', None)
            rel_is_part_name = relations_json.get('isPartName', None)
            rel_is_disambiguation = relations_json.get('isDisambiguation', None)
            rel_disambiguation = relations_json.get('disambiguationLinks', None)
            rel_disambiguation_norm = relations_json.get('disambiguationLinksNorm', None)
            rel_parenthesis = relations_json.get('titleParenthesis', None)
            rel_parenthesis_norm = relations_json.get('titleParenthesisNorm', None)
            rel_categories = relations_json.get('categories', None)
            rel_categories_norm = relations_json.get('categoriesNorm', None)
            rel_be_comp = relations_json.get('beCompRelations', None)
            rel_be_comp_norm = relations_json.get('beCompRelationsNorm', None)
            rel_aliases = relations_json.get('aliases', None)
            rel_aliases_norm = relations_json.get('aliasesNorm', None)

            relations = WikipediaPageExtractedRelations(rel_is_part_name, rel_is_disambiguation,
                                                        rel_parenthesis,
                                                        rel_disambiguation,
                                                        rel_categories, rel_aliases, rel_be_comp,
                                                        rel_disambiguation_norm,
                                                        rel_categories_norm, rel_aliases_norm,
                                                        rel_parenthesis_norm,
                                                        rel_be_comp_norm)

            page = WikipediaPage(orig_phrase, orig_phrase_norm, wiki_title, wiki_title_norm, 0,
                                 pageid, description,
                                 relations)
            pages.add(WikipediaSearchPageResult(orig_phrase, page))

        return pages

    def load_dump(self, wiki_dump):
        onlyfiles = []
        for _file in listdir(wiki_dump):
            file_path = join(wiki_dump, _file)
            if isfile(file_path):
                onlyfiles.append(file_path)

        json_dump_list = {}
        for _file in onlyfiles:
            json_dump_list.update(load_json_file(_file))

        dump_final = {}
        for key, value in json_dump_list.items():
            dump_final[key] = self.extract_json_values(value)

        return dump_final

    class NoPage(object):
        """ Attribute not found. """

        def __init__(self, *args, **kwargs):  # real signature unknown
            pass

        @staticmethod  # known case of __new__
        def __new__(S, *more):  # real signature unknown; restored from __doc__
            """ T.__new__(S, ...) -> a new object with type S, a subtype of T """
