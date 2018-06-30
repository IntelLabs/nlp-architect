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
from __future__ import unicode_literals, print_function, division, \
    absolute_import
import json


class HighLevelDoc:
    """
    object for annotation documents

    Args:
        self.doc_text (str): document text
        self.annotation_set (list(str)): list of all annotations in doc
        self.spans (list(dict)): list of span dict, each span_dict is structured as follows:
            { 'end': (int), 'start': (int), 'type': (str) string of annotation }
    """
    def __init__(self):
        self.doc_text = None
        self.annotation_set = []
        self.spans = []

    def json(self):
        """
        Return json representations of the object

        Returns:
            :obj:`json`: json representations of the object
        """
        return json.dumps(self.__dict__)

    def pretty_json(self):
        """
        Return pretty json representations of the object

        Returns:
            :obj:`json`: pretty json representations of the object
        """
        return json.dumps(self.__dict__, indent=4)

    def displacy_doc(self):  # only change annotations to lowercase
        """
        Return doc adapted to displacyENT expected input
        """
        self.annotation_set = [annotation.lower() for annotation in self.annotation_set]
        return self.__dict__
