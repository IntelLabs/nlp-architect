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
import os

from nlp_architect.common.cdc.cluster import Clusters
from nlp_architect.models.cross_doc_coref.system.cdc_settings import CDCSettings
from nlp_architect.models.cross_doc_coref.system.cdc_utils import write_clusters_to_file, \
    write_event_coref_scorer_results, write_entity_coref_scorer_results
from nlp_architect.models.cross_doc_coref.system.sieves.run_sieve_system import RunSystemsEvent, \
    RunSystemsEntity
from nlp_architect.utils import io

logger = logging.getLogger(__name__)


def run_event_coref(resources: CDCSettings) -> Clusters:
    """
    Running Cross Document Coref on event mentions
    Args:
        resources: resources for running the evaluation

    Returns:
        Clusters: List of clusters and mentions with predicted cross doc coref within each topic
    """
    io.create_folder(resources.cdc_resources.eval_output_dir)
    for topic in resources.events_topics.topics_list:
        sieves_list_event = RunSystemsEvent(topic, resources)
        clusters = sieves_list_event.run_deterministic()

        clusters.set_coref_chain_to_mentions()

        with open(os.path.join(
                resources.cdc_resources.eval_output_dir, 'event_clusters.txt'), 'w') \
                as event_clusters_file:
            write_clusters_to_file(clusters, topic.topic_id, event_clusters_file)

    logger.info('Write event coref results')
    write_event_coref_scorer_results(resources.events_topics.topics_list,
                                     resources.cdc_resources.eval_output_dir)
    return clusters


def run_entity_coref(resources: CDCSettings) -> Clusters:
    """
    Running Cross Document Coref on Entity mentions
    Args:
        resources: (CDCSettings) resources for running the evaluation

    Returns:
        Clusters: List of topics and mentions with predicted cross doc coref within each topic
    """
    io.create_folder(resources.cdc_resources.eval_output_dir)
    for topic in resources.entity_topics.topics_list:
        sieves_list_entity = RunSystemsEntity(topic, resources)
        clusters = sieves_list_entity.run_deterministic()

        clusters.set_coref_chain_to_mentions()

        with open(os.path.join(
                resources.cdc_resources.eval_output_dir, 'entity_clusters.txt'), 'w') \
                as entity_clusters_file:
            write_clusters_to_file(clusters, topic.topic_id, entity_clusters_file)

    logger.info('Write entity coref results')
    write_entity_coref_scorer_results(resources.entity_topics.topics_list,
                                      resources.cdc_resources.eval_output_dir)

    return clusters
