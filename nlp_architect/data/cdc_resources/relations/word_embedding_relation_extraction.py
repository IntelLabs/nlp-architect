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
import math
from typing import List, Set

from scipy.spatial.distance import cosine as cos

from nlp_architect.common.cdc.mention_data import MentionDataLight
from nlp_architect.data.cdc_resources.embedding.embed_elmo import ElmoEmbedding, \
    ElmoEmbeddingOffline
from nlp_architect.data.cdc_resources.embedding.embed_glove import GloveEmbedding, \
    GloveEmbeddingOffline
from nlp_architect.data.cdc_resources.relations.relation_extraction import RelationExtraction
from nlp_architect.data.cdc_resources.relations.relation_types_enums import EmbeddingMethod, \
    RelationType
from nlp_architect.utils.string_utils import StringUtils

logger = logging.getLogger(__name__)


class WordEmbeddingRelationExtraction(RelationExtraction):
    def __init__(self, method: EmbeddingMethod = EmbeddingMethod.GLOVE,
                 glove_file: str = None, elmo_file: str = None, cos_accepted_dist: float = 0.7):
        """
        Extract Relation between two mentions according to Word Embedding cosine distance

        Args:
            method (optional): EmbeddingMethod.{GLOVE/GLOVE_OFFLINE/ELMO/ELMO_OFFLINE}
                (default = GLOVE)
            glove_file (required on GLOVE/GLOVE_OFFLINE mode): str Location of Glove file
            elmo_file (required on ELMO_OFFLINE mode): str Location of Elmo file
        """
        if method == EmbeddingMethod.GLOVE:
            self.embedding = GloveEmbedding(glove_file)
            self.contextual = False
        elif method == EmbeddingMethod.GLOVE_OFFLINE:
            self.embedding = GloveEmbeddingOffline(glove_file)
            self.contextual = False
        elif method == EmbeddingMethod.ELMO:
            self.embedding = ElmoEmbedding()
            self.contextual = True
        elif method == EmbeddingMethod.ELMO_OFFLINE:
            self.embedding = ElmoEmbeddingOffline(elmo_file)
            self.contextual = True

        self.accepted_dist = cos_accepted_dist
        super(WordEmbeddingRelationExtraction, self).__init__()

    def extract_all_relations(self, mention_x: MentionDataLight,
                              mention_y: MentionDataLight) -> Set[RelationType]:
        ret_ = set()
        ret_.add(self.extract_sub_relations(mention_x, mention_y,
                                            RelationType.WORD_EMBEDDING_MATCH))
        return ret_

    def extract_sub_relations(self, mention_x: MentionDataLight, mention_y: MentionDataLight,
                              relation: RelationType) -> RelationType:
        """
        Check if input mentions has the given relation between them

        Args:
            mention_x: MentionDataLight
            mention_y: MentionDataLight
            relation: RelationType

        Returns:
            RelationType: relation in case mentions has given relation or
                RelationType.NO_RELATION_FOUND otherwise
        """
        if relation is not RelationType.WORD_EMBEDDING_MATCH:
            return RelationType.NO_RELATION_FOUND

        mention_x_str = mention_x.tokens_str
        mention_y_str = mention_y.tokens_str
        if StringUtils.is_pronoun(mention_x_str.lower()) or \
                StringUtils.is_pronoun(mention_y_str.lower()):
            if not self.contextual:
                return RelationType.NO_RELATION_FOUND

            if mention_x.mention_context is None or mention_y.mention_context is None:
                return RelationType.NO_RELATION_FOUND

        if self.is_word_embed_match(mention_x, mention_y):
            return RelationType.WORD_EMBEDDING_MATCH

        return RelationType.NO_RELATION_FOUND

    def is_word_embed_match(self, mention_x: MentionDataLight, mention_y: MentionDataLight):
        """
        Check if input mentions Word Embedding cosine distance below above 0.65

        Args:
            mention_x: MentionDataLight
            mention_y: MentionDataLight

        Returns:
            bool
        """
        match_result = False
        x_embed = self.embedding.get_head_feature_vector(mention_x)
        y_embed = self.embedding.get_head_feature_vector(mention_y)
        # make sure words are not 'unk/None/0'
        if x_embed is not None and y_embed is not None:
            dist = cos(x_embed, y_embed)
            if not math.isnan(dist):
                sim = 1 - dist
                if sim >= self.accepted_dist:
                    match_result = True

        return match_result

    @staticmethod
    def get_supported_relations() -> List[RelationType]:
        """
        Return all supported relations by this class

        Returns:
            List[RelationType]
        """
        return [RelationType.WORD_EMBEDDING_MATCH]
