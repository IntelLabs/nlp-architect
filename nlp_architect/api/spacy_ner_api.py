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
from nlp_architect.utils.high_level_doc import HighLevelDoc
from nlp_architect.utils.text import SpacyInstance


class SpacyNerApi(AbstractApi):
    """
    Spacy NER model API
    """
    def __init__(self):
        self.model = None

    def load_model(self):
        """
        Load spacy english model
        """
        self.model = SpacyInstance(disable=['parser', 'vectors', 'textcat']).parser

    def inference(self, doc):
        """
        Parse according to SpacyNer's model

        Args:
            doc (str): the doc str

        Returns:
            :obj:`nlp_architect.utils.high_level_doc.HighLevelDoc`: the model's response hosted in
                HighLevelDoc object
        """
        spacy_doc = self.model(doc)
        ents = \
            [{"start": e.start_char, "end": e.end_char, "type": e.label_} for e in spacy_doc.ents]
        annot_doc = HighLevelDoc()
        annot_doc.doc_text = doc
        annot_doc.annotation_set = [e.label_ for e in spacy_doc.ents]
        annot_doc.spans = ents
        return annot_doc




