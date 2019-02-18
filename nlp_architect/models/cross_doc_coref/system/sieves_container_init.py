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
from typing import List

from nlp_architect.data.cdc_resources.relations.relation_extraction import RelationExtraction
from nlp_architect.models.cross_doc_coref.sieves_config import \
    EventSievesConfiguration, EntitySievesConfiguration

logger = logging.getLogger(__name__)


class SievesContainerInitialization(object):
    def __init__(self, event_coref_config: EventSievesConfiguration,
                 entity_coref_config: EntitySievesConfiguration,
                 sieves_model_list: List[RelationExtraction]):
        self.sieves_model_list = sieves_model_list
        self.event_config = event_coref_config
        self.entity_config = entity_coref_config

    def get_module_from_relation(self, relation_type):
        for model in self.sieves_model_list:
            if relation_type in model.get_supported_relations():
                return model

        raise Exception('No model found that Support RelationType-' + str(relation_type))
