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

import traceback

import requests
from elasticsearch import Elasticsearch

from nlp_architect.data.cdc_resources.data_types.wiki.wikipedia_page import WikipediaPage
from nlp_architect.data.cdc_resources.data_types.wiki.wikipedia_page_extracted_relations import \
    WikipediaPageExtractedRelations
from nlp_architect.data.cdc_resources.wikipedia.wiki_search_page_result import \
    WikipediaSearchPageResult


class WikiElastic(object):
    def __init__(self, host: str, port: int, index: str):
        # connect to our cluster
        self.cache = dict()
        if self.is_connected(host, port):
            self.es_index = index
            self.es = Elasticsearch([{'host': host, 'port': port}])
        else:
            traceback.print_exc()
            raise IOError('Cannot connect to ElasticSearch node')

    @staticmethod
    def is_connected(elastic_host, elastic_port):
        elastic_search_url = 'http://' + elastic_host + ':' + str(elastic_port)
        res = requests.get(elastic_search_url)
        if res.content:
            return True
        return False

    def get_pages(self, phrase):
        if phrase in self.cache:
            return self.cache[phrase]
        try:
            phrase_strip = ' '.join(phrase.replace('-', ' ').split())
            pages = set()
            best_results = self.get_best_elastic_results(phrase_strip)
            for result in best_results:
                _id = result['_id']
                if _id != 0:
                    result_source = result['_source']
                    if 'redirectTitle' in result_source:
                        redirect_title = result_source['redirectTitle']
                        red_result = None
                        while redirect_title and result_source['title'] != redirect_title:
                            red_result = self.get_redirect_result(redirect_title)
                            if red_result is None or len(red_result) == 0:
                                print('could not find redirect title=' + redirect_title
                                      + ', does not exist in data')
                                redirect_title = None
                            elif 'redirectTitle' in red_result[0]['_source']:
                                redirect_title = red_result[0]['_source']['redirectTitle']
                            else:
                                redirect_title = None

                        if red_result is not None and len(red_result) > 0:
                            result = red_result[0]
                            _id = result['_id']

                    elastic_page_result = self.get_page_from_result_v1(phrase_strip, result, _id)
                    pages.add(WikipediaSearchPageResult(phrase, elastic_page_result))

            self.cache[phrase] = pages
            return pages
        except Exception:
            traceback.print_exc()

    def get_best_elastic_results(self, phrase):
        best_results = []
        best_results.extend(self.get_redirect_result(phrase))

        search_result_near_match = self.es.search(index=self.es_index,
                                                  body={"size": 5,
                                                        'query': {'match_phrase': {
                                                            'title.near_match': phrase}}})
        best_results.extend(self.extract_from_elastic_results(search_result_near_match))

        return best_results

    @staticmethod
    def extract_from_elastic_results(search_result):
        best_results = []
        if search_result is not None and search_result['hits']['total'] > 0:
            if search_result['hits']['total'] > 0:
                best_results = search_result['hits']['hits']
        return best_results

    def get_redirect_result(self, phrase):
        search_result = self.es.search(index=self.es_index,
                                       body={"size": 5,
                                             'query': {'match_phrase': {'title.keyword': phrase}}})
        results = self.extract_from_elastic_results(search_result)
        return results

    def get_page_from_result_v1(self, phrase, result, result_id):
        if result_id != 0 and result is not None:
            relations = None
            result_source = result['_source']
            result_score = result['_score']
            if result_source is not None:
                title = result_source['title']
                relations_source = result_source['relations']

                if relations_source is not None:
                    is_part = relations_source['isPartName']
                    is_disambig = relations_source['isDisambiguation']

                    disambig_links = self.safe_extract_field_from_dict('disambiguationLinks',
                                                                       relations_source)
                    disambig_links_norm = self.safe_extract_field_from_dict(
                        'disambiguationLinksNorm', relations_source)
                    categories = self.safe_extract_field_from_dict('categories', relations_source)
                    categories_norm = self.safe_extract_field_from_dict('categoriesNorm',
                                                                        relations_source)
                    title_parent = self.safe_extract_field_from_dict('titleParenthesis',
                                                                     relations_source)
                    title_parent_norm = self.safe_extract_field_from_dict('titleParenthesisNorm',
                                                                          relations_source)
                    be_comp = self.safe_extract_field_from_dict('beCompRelations',
                                                                relations_source)
                    be_comp_norm = self.safe_extract_field_from_dict('beCompRelationsNorm',
                                                                     relations_source)

                    relations = WikipediaPageExtractedRelations(
                        is_part, is_disambig, title_parent, disambig_links,
                        categories, None, be_comp, disambig_links_norm, categories_norm, None,
                        title_parent_norm,
                        be_comp_norm
                    )

            return WikipediaPage(phrase, None, title, None, result_score, result_id, None,
                                 relations)

        return WikipediaPage()

    @staticmethod
    def safe_extract_field_from_dict(field_name, _dict):
        if field_name in _dict:
            return _dict[field_name]
        return None
