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

from nlp_architect.api.abstract_api import AbstractApi
from nlp_architect.pipelines.spacy_bist import SpacyBISTParser


class BistParserApi(AbstractApi):
    """
    Bist Parser API
    """
    def __init__(self):
        self.model = None

    def load_model(self):
        """
        Load SpacyBISTParser model
        """
        self.model = SpacyBISTParser()

    def inference(self, doc):
        """
        Parse according to SpacyBISTParser's model

        Args:
            doc (str): the doc str

        Returns:
            CoreNLPDoc: the parser's response hosted in CoreNLPDoc object
        """
        return self.model.parse(doc)
