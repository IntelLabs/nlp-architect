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
from nlp_architect.models.cross_doc_coref.system.sieves.sieves import SieveType


class CDCConfig(object):
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
    def sieves_order(self, sieves_order: List[Tuple[SieveType, RelationType, float]]):
        self.__sieves_order = sieves_order

    @property
    def run_evaluation(self):
        """Should run evaluation (True/False)"""
        return self.__run_evaluation

    @run_evaluation.setter
    def run_evaluation(self, run_evaluation: bool):
        self.__run_evaluation = run_evaluation


class EventConfig(CDCConfig):
    def __init__(self):
        super(EventConfig, self).__init__()

        self.run_evaluation = True

        self.sieves_order = [
            (SieveType.STRICT, RelationType.SAME_HEAD_LEMMA, 0.0),
            (SieveType.STRICT, RelationType.EXACT_STRING, 0.0),
            (SieveType.VERY_RELAX, RelationType.WIKIPEDIA_DISAMBIGUATION, 0.1),
            (SieveType.VERY_RELAX, RelationType.WORD_EMBEDDING_MATCH, 0.7),
            (SieveType.VERY_RELAX, RelationType.WIKIPEDIA_REDIRECT_LINK, 0.1),
            (SieveType.RELAX, RelationType.FUZZY_HEAD_FIT, 0.5),
            (SieveType.STRICT, RelationType.FUZZY_FIT, 0.0),
            (SieveType.RELAX, RelationType.SAME_HEAD_LEMMA_RELAX, 0.5),
            (SieveType.STRICT, RelationType.WITHIN_DOC_COREF, 0.0),
            (SieveType.VERY_RELAX, RelationType.WIKIPEDIA_TITLE_PARENTHESIS, 0.1),
            (SieveType.VERY_RELAX, RelationType.WIKIPEDIA_BE_COMP, 0.1),
            (SieveType.VERY_RELAX, RelationType.WIKIPEDIA_CATEGORY, 0.1),
            (SieveType.VERY_RELAX, RelationType.VERBOCEAN_MATCH, 0.1),
            (SieveType.STRICT, RelationType.WORDNET_SAME_SYNSET_EVENT, 0.0),
            (SieveType.STRICT, RelationType.WORDNET_DERIVATIONALLY, 0.0)
        ]


class EntityConfig(CDCConfig):
    def __init__(self):
        super(EntityConfig, self).__init__()

        self.run_evaluation = True

        self.sieves_order = [
            (SieveType.STRICT, RelationType.SAME_HEAD_LEMMA, 0.0),
            (SieveType.STRICT, RelationType.EXACT_STRING, 0.0),
            (SieveType.STRICT, RelationType.FUZZY_FIT, 0.0),
            (SieveType.VERY_RELAX, RelationType.WIKIPEDIA_REDIRECT_LINK, 0.1),
            (SieveType.VERY_RELAX, RelationType.WIKIPEDIA_DISAMBIGUATION, 0.1),
            (SieveType.VERY_RELAX, RelationType.WORD_EMBEDDING_MATCH, 0.7),
            (SieveType.VERY_RELAX, RelationType.WORDNET_PARTIAL_SYNSET_MATCH, 0.1),
            (SieveType.RELAX, RelationType.FUZZY_HEAD_FIT, 0.5),
            (SieveType.VERY_RELAX, RelationType.WIKIPEDIA_CATEGORY, 0.1),
            (SieveType.STRICT, RelationType.WITHIN_DOC_COREF, 0.0),
            (SieveType.VERY_RELAX, RelationType.WIKIPEDIA_BE_COMP, 0.1),
            (SieveType.RELAX, RelationType.SAME_HEAD_LEMMA_RELAX, 0.5),
            (SieveType.VERY_RELAX, RelationType.WIKIPEDIA_TITLE_PARENTHESIS, 0.1),
            (SieveType.STRICT, RelationType.WORDNET_SAME_SYNSET_ENTITY, 0.0),
            (SieveType.VERY_RELAX, RelationType.REFERENT_DICT, 0.5)
        ]
