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
from typing import List, Tuple

from nlp_architect.data.cdc_resources.relations.relation_types_enums import RelationType


class SievesConfiguration(object):
    def __init__(self):
        """Cross document co-reference event and entity evaluation configuration settings"""

        self.__sieves_order = None
        self.__run_evaluation = False

    @property
    def sieves_order(self):
        """
        Sieve definition and Sieve running order

        Tuple[SieveType, RelationType, Threshold(float)] - define sieves to run, were

        Strict- Merge clusters only in case all mentions has current relation between them,
        Relax- Merge clusters in case (matched mentions) / len(cluster_1.mentions)) >= thresh,
        Very_Relax- Merge clusters in case (matched mentions) / (all possible pairs) >= thresh

        RelationType represent the type of sieve to run.

        """
        return self.__sieves_order

    @sieves_order.setter
    def sieves_order(self, sieves_order: List[Tuple[RelationType, float]]):
        self.__sieves_order = sieves_order

    @property
    def run_evaluation(self):
        """Should run evaluation (True/False)"""
        return self.__run_evaluation

    @run_evaluation.setter
    def run_evaluation(self, run_evaluation: bool):
        self.__run_evaluation = run_evaluation


class EventSievesConfiguration(SievesConfiguration):
    def __init__(self):
        super(EventSievesConfiguration, self).__init__()

        self.run_evaluation = True

        self.sieves_order = [
            (RelationType.SAME_HEAD_LEMMA, 1.0),
            (RelationType.EXACT_STRING, 1.0),
            (RelationType.WIKIPEDIA_DISAMBIGUATION, 0.1),
            (RelationType.WORD_EMBEDDING_MATCH, 0.7),
            (RelationType.WIKIPEDIA_REDIRECT_LINK, 0.1),
            (RelationType.FUZZY_HEAD_FIT, 0.5),
            (RelationType.FUZZY_FIT, 1.0),
            (RelationType.WITHIN_DOC_COREF, 1.0),
            (RelationType.WIKIPEDIA_TITLE_PARENTHESIS, 0.1),
            (RelationType.WIKIPEDIA_BE_COMP, 0.1),
            (RelationType.WIKIPEDIA_CATEGORY, 0.1),
            (RelationType.VERBOCEAN_MATCH, 0.1),
            (RelationType.WORDNET_DERIVATIONALLY, 1.0)
        ]


class EntitySievesConfiguration(SievesConfiguration):
    def __init__(self):
        super(EntitySievesConfiguration, self).__init__()

        self.run_evaluation = True

        self.sieves_order = [
            (RelationType.SAME_HEAD_LEMMA, 1.0),
            (RelationType.EXACT_STRING, 1.0),
            (RelationType.FUZZY_FIT, 1.0),
            (RelationType.WIKIPEDIA_REDIRECT_LINK, 0.1),
            (RelationType.WIKIPEDIA_DISAMBIGUATION, 0.1),
            (RelationType.WORD_EMBEDDING_MATCH, 0.7),
            (RelationType.WORDNET_PARTIAL_SYNSET_MATCH, 0.1),
            (RelationType.FUZZY_HEAD_FIT, 0.5),
            (RelationType.WIKIPEDIA_CATEGORY, 0.1),
            (RelationType.WITHIN_DOC_COREF, 1.0),
            (RelationType.WIKIPEDIA_BE_COMP, 0.1),
            (RelationType.WIKIPEDIA_TITLE_PARENTHESIS, 0.1),
            (RelationType.WORDNET_SAME_SYNSET, 1.0),
            (RelationType.REFERENT_DICT, 0.5)
        ]
