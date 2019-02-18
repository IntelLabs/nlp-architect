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
import time

from nlp_architect.common.cdc.cluster import Clusters
from nlp_architect.common.cdc.topics import Topic
from nlp_architect.models.cross_doc_coref.system.sieves.sieves import SieveClusterMerger
from nlp_architect.models.cross_doc_coref.system.sieves_container_init import \
    SievesContainerInitialization

logger = logging.getLogger(__name__)


class RunSystemsSuper(object):
    def __init__(self, topic: Topic):
        self.sieves = []
        self.results_dict = dict()
        self.results_ordered = []
        logger.info('loading topic %s, total mentions: %d', topic.topic_id, len(topic.mentions))
        self.clusters = Clusters(topic.topic_id, topic.mentions)

    @staticmethod
    def set_sieves_from_config(config, get_rel_extraction):
        sieves = []
        for _type_tup in config.sieves_order:
            sieves.append(SieveClusterMerger(_type_tup, get_rel_extraction(_type_tup[0])))
        return sieves

    def run_deterministic(self):
        # pylint: disable=too-many-nested-blocks
        for sieve in self.sieves:
            start = time.time()
            clusters_changed = True
            merge_count = 0
            while clusters_changed:
                clusters_changed = False
                clusters_size = len(self.clusters.clusters_list)
                for i in range(0, clusters_size):
                    cluster_i = self.clusters.clusters_list[i]
                    if cluster_i.merged:
                        continue

                    for j in range(i + 1, clusters_size):
                        cluster_j = self.clusters.clusters_list[j]
                        if cluster_j.merged:
                            continue

                        if cluster_i is not cluster_j:
                            criterion = sieve.run_sieve(cluster_i, cluster_j)
                            if criterion:
                                merge_count += 1
                                clusters_changed = True
                                cluster_i.merge_clusters(cluster_j)
                                cluster_j.merged = True

                if clusters_changed:
                    self.clusters.clean_clusters()

            end = time.time()
            took = end - start
            logger.info('Total of %d clusters merged using method: %s, took: %.4f sec',
                        merge_count, str(sieve.excepted_relation), took)

        return self.clusters

    def get_results(self):
        return self.results_ordered


class RunSystemsEntity(RunSystemsSuper):
    def __init__(self, topic: Topic, resources):
        super(RunSystemsEntity, self).__init__(topic)
        self.sieves = self.set_sieves_from_config(resources.entity_config,
                                                  resources.get_module_from_relation)


class RunSystemsEvent(RunSystemsSuper):
    def __init__(self, topic, resources):
        super(RunSystemsEvent, self).__init__(topic)
        self.sieves = self.set_sieves_from_config(resources.event_config,
                                                  resources.get_module_from_relation)


# pylint: disable=no-else-return
def get_run_system(topic: Topic, resource: SievesContainerInitialization, eval_type: str):
    if eval_type.lower() == 'entity':
        return RunSystemsEntity(topic, resource)
    if eval_type.lower() == 'event':
        return RunSystemsEvent(topic, resource)
    else:
        raise AttributeError(eval_type + ' Not supported!')
