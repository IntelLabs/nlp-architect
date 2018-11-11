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

from nlp_architect import LIBRARY_ROOT
from nlp_architect.data.cdc_resources.relations.relation_types_enums import RelationType
from nlp_architect.models.cross_doc_coref.cdc_config import EventConfig, EntityConfig
from nlp_architect.models.cross_doc_coref.cdc_resource import CDCResources
from nlp_architect.models.cross_doc_coref.system.cdc_settings import CDCSettings
from nlp_architect.models.cross_doc_coref.system.sieves.sieves import SieveType
from nlp_architect.models.cross_doc_sieves import run_event_coref, run_entity_coref


def run_example():
    event_config = EventConfig()
    event_config.sieves_order = [
        (SieveType.STRICT, RelationType.SAME_HEAD_LEMMA, 0.0),
        (SieveType.VERY_RELAX, RelationType.WIKIPEDIA_DISAMBIGUATION, 0.1),
        (SieveType.VERY_RELAX, RelationType.WORD_EMBEDDING_MATCH, 0.7),
        (SieveType.RELAX, RelationType.SAME_HEAD_LEMMA_RELAX, 0.5),
    ]

    event_config.gold_mentions_file = LIBRARY_ROOT + \
        '/datasets/ecb/ecb_all_event_mentions.json'

    entity_config = EntityConfig()

    entity_config.sieves_order = [
        (SieveType.STRICT, RelationType.SAME_HEAD_LEMMA, 0.0),
        (SieveType.VERY_RELAX, RelationType.WIKIPEDIA_REDIRECT_LINK, 0.1),
        (SieveType.VERY_RELAX, RelationType.WIKIPEDIA_DISAMBIGUATION, 0.1),
        (SieveType.VERY_RELAX, RelationType.WORD_EMBEDDING_MATCH, 0.7),
        (SieveType.VERY_RELAX, RelationType.REFERENT_DICT, 0.5)
    ]

    entity_config.gold_mentions_file = LIBRARY_ROOT + \
        '/datasets/ecb/ecb_all_entity_mentions.json'

    # CDCResources hold default attribute values that might need to be change,
    # (using the defaults values in this example), use to configure attributes
    # such as resources files location, output directory, resources init methods and other.
    # check in class and see if any attributes require change in your set-up
    resource_location = CDCResources()
    resources = CDCSettings(resource_location, event_config, entity_config)

    event_clusters = None
    if event_config.run_evaluation:
        logger.info('Running event coreference resolution')
        event_clusters = run_event_coref(resources)

    entity_clusters = None
    if entity_config.run_evaluation:
        logger.info('Running entity coreference resolution')
        entity_clusters = run_entity_coref(resources)

    print('-=Cross Document Coref Results=-')
    print('-=Event Clusters Mentions=-')
    for event_cluster in event_clusters.clusters_list:
        print(event_cluster.coref_chain)
        for event_mention in event_cluster.mentions:
            print(event_mention.mention_id)
            print(event_mention.tokens_str)

    print('-=Entity Clusters Mentions=-')
    for entity_cluster in entity_clusters.clusters_list:
        print(entity_cluster.coref_chain)
        for entity_mention in entity_cluster.mentions:
            print(entity_mention.mention_id)
            print(entity_mention.tokens_str)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    run_example()
