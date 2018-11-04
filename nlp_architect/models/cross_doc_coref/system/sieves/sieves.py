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
from enum import Enum
from typing import Tuple

from nlp_architect.common.cdc.cluster import Cluster
from nlp_architect.data.cdc_resources.relations.relation_extraction import RelationExtraction
from nlp_architect.data.cdc_resources.relations.relation_types_enums import RelationType

logger = logging.getLogger(__name__)


class SieveType(Enum):
    STRICT = 1
    RELAX = 2
    VERY_RELAX = 3


class SieveStrict(object):
    def __init__(self, excepted_relation: Tuple[SieveType, RelationType, float],
                 relation_extractor: RelationExtraction):
        """

        Args:
            excepted_relation: tuple with relation to run in sieve, threshold to merge clusters
            relation_extractor:
        """
        self.excepted_relation = excepted_relation[1]
        self.threshold = excepted_relation[2]
        self.relation_extractor = relation_extractor

        logger.info('init %s Sieve, for relation-%s with threshold=%.1f',
                    excepted_relation[0].name, self.excepted_relation.name, self.threshold)

    def run_sieve(self, cluster_i: Cluster, cluster_j: Cluster):
        """

        Args:
            cluster_i:
            cluster_j:

        Returns:

        """
        for mention_i in cluster_i.mentions:
            for mention_j in cluster_j.mentions:
                if RelationType.NO_RELATION_FOUND == self.relation_extractor.extract_sub_relations(
                        mention_i, mention_j, self.excepted_relation):
                    return False

        return True


class SieveRelax(SieveStrict):
    def __init__(self, excepted_relation: Tuple[RelationType, float],
                 relation_extractor: RelationExtraction):
        super(SieveRelax, self).__init__(excepted_relation, relation_extractor)

    def run_sieve(self, cluster_i: Cluster, cluster_j: Cluster):
        """

        Args:
            cluster_i:
            cluster_j:

        Returns:

        """
        matches = 0
        for mention_i in cluster_i.mentions:
            for mention_j in cluster_j.mentions:
                match_result = self.relation_extractor.extract_sub_relations(
                    mention_i, mention_j, self.excepted_relation)

                if match_result == self.excepted_relation:
                    matches += 1
                    break

        matches_rate = matches / float(len(cluster_i.mentions))

        result = False
        if matches_rate >= self.threshold:
            result = True

        return result


class SieveVeryRelaxed(SieveStrict):
    def __init__(self, excepted_relation: Tuple[RelationType, float],
                 relation_extractor: RelationExtraction):
        super(SieveVeryRelaxed, self).__init__(excepted_relation, relation_extractor)

    def run_sieve(self, cluster_i: Cluster, cluster_j: Cluster):
        """

        Args:
            cluster_i:
            cluster_j:

        Returns:

        """
        matches = 0
        for mention_i in cluster_i.mentions:
            for mention_j in cluster_j.mentions:
                match_result = self.relation_extractor.extract_sub_relations(
                    mention_i, mention_j, self.excepted_relation)
                if match_result == self.excepted_relation:
                    matches += 1

        possible_pairs_len = float(len(cluster_i.mentions) * len(cluster_j.mentions))
        matches_rate = matches / possible_pairs_len

        result = False
        if matches_rate >= self.threshold:
            result = True

        return result


def get_sieve(relation_tup, rel_extractor):
    sieve = None
    if relation_tup[0] == SieveType.STRICT:
        sieve = SieveStrict(relation_tup, rel_extractor)
    elif relation_tup[0] == SieveType.RELAX:
        sieve = SieveRelax(relation_tup, rel_extractor)
    elif relation_tup[0] == SieveType.VERY_RELAX:
        sieve = SieveVeryRelaxed(relation_tup, rel_extractor)
    else:
        raise Exception('Not supported SieveType')

    return sieve
