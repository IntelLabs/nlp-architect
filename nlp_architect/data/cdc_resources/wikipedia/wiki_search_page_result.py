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
from nlp_architect.data.cdc_resources.data_types.wiki.wikipedia_page import WikipediaPage


class WikipediaSearchPageResult(object):
    def __init__(self, search_phrase: str, page_result: WikipediaPage):
        """
        Args:
            search_phrase: the search phrase that yield this page result
            page_result: page result for search phrase
        """
        self.search_phrase = search_phrase
        self.page_result = page_result
