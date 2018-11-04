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

from nlp_architect.data.cdc_resources.data_types.wn.wordnet_page import WordnetPage
from nlp_architect.utils.io import load_json_file

logger = logging.getLogger(__name__)


class WordnetOffline(object):
    def __init__(self, wordnet_dump):
        if wordnet_dump:
            self.dump = self.load_dump(wordnet_dump)
            logger.info('Wikipedia dump loaded successfully!')

    def get_pages(self, mention):
        page = None
        if mention.tokens_str is not None and mention.tokens_str in self.dump:
            page = self.dump[mention.tokens_str]

        return page

    def load_dump(self, wn_dump):
        onlyfiles = []
        for _file in listdir(wn_dump):
            file_path = join(wn_dump, _file)
            if isfile(file_path):
                onlyfiles.append(file_path)

        json_dump_list = {}
        for _file in onlyfiles:
            json_dump_list.update(load_json_file(_file))

        dump_final = {}
        for key, value in json_dump_list.items():
            dump_final[key] = self.extract_json_values(value)

        return dump_final

    @staticmethod
    def extract_json_values(json_page):
        if json_page is not None:
            orig_phrase = json_page.get('orig_phrase', None)
            clean_phrase = json_page.get('clean_phrase', None)
            head = json_page.get('head', None)
            head_lemma = json_page.get('head_lemma', None)
            head_synonyms = set(json_page.get('head_synonyms', None))
            head_lemma_synonyms = set(json_page.get('head_lemma_synonyms', None))
            head_derivationally = set(json_page.get('head_derivationally', None))
            head_lemma_derivationally = set(json_page.get('head_lemma_derivationally', None))

            all_clean_words_synonyms = json_page.get('all_clean_words_synonyms', None)
            all_as_set_list = []
            for list_ in all_clean_words_synonyms:
                all_as_set_list.append(set(list_))

            wordnet_page = WordnetPage(orig_phrase, clean_phrase, head,
                                       head_lemma, head_synonyms,
                                       head_lemma_synonyms, head_derivationally,
                                       head_lemma_derivationally,
                                       all_as_set_list)
            return wordnet_page

        return None
