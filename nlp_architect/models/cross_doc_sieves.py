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

from nlp_architect.common.cdc.cluster import Clusters
from nlp_architect.common.cdc.topics import Topics
from nlp_architect.models.cross_doc_coref.system.sieves.run_sieve_system import get_run_system
from nlp_architect.models.cross_doc_coref.system.sieves_container_init import \
    SievesContainerInitialization

logger = logging.getLogger(__name__)


def run_event_coref(topics: Topics, resources: SievesContainerInitialization) -> List[Clusters]:
    """
    Running Cross Document Coref on event mentions
    Args:
        topics   : The Topics (with mentions) to evaluate
        resources: resources for running the evaluation

    Returns:
        Clusters: List of clusters and mentions with predicted cross doc coref within each topic
    """

    return _run_coref(topics, resources, 'event')


def run_entity_coref(topics: Topics, resources: SievesContainerInitialization) -> List[Clusters]:
    """
    Running Cross Document Coref on Entity mentions
    Args:
        topics   : The Topics (with mentions) to evaluate
        resources: (SievesContainerInitialization) resources for running the evaluation

    Returns:
        Clusters: List of topics and mentions with predicted cross doc coref within each topic
    """
    return _run_coref(topics, resources, 'entity')


def _run_coref(topics: Topics, resources: SievesContainerInitialization,
               eval_type: str) -> List[Clusters]:
    """
    Running Cross Document Coref on Entity mentions
    Args:
        resources: (SievesContainerInitialization) resources for running the evaluation
        topics   : The Topics (with mentions) to evaluate

    Returns:
        Clusters: List of topics and mentions with predicted cross doc coref within each topic
    """
    clusters_list = list()
    for topic in topics.topics_list:
        sieves_list = get_run_system(topic, resources, eval_type)
        clusters = sieves_list.run_deterministic()
        clusters.set_coref_chain_to_mentions()
        clusters_list.append(clusters)

    return clusters_list
