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
from typing import List, Set

from nlp_architect.common.cdc.mention_data import MentionData
from nlp_architect.data.cdc_resources.relations.relation_extraction import RelationExtraction
from nlp_architect.data.cdc_resources.relations.relation_types_enums import RelationType
from nlp_architect.utils.io import load_json_file

logger = logging.getLogger(__name__)


class WithinDocCoref(RelationExtraction):
    def __init__(self, wd_file: str):
        """
        Extract Relation between two mentions according to Within document co-reference

        Args:
            wd_file (required): str Location of within doc co-reference mentions file
        """
        logger.info('Loading Within doc resource')
        if wd_file is not None and os.path.isfile(wd_file):
            wd_mentions_json = load_json_file(wd_file)
            self.within_doc_coref_chain = self.arrange_resource(wd_mentions_json)
        else:
            raise FileNotFoundError('Within-doc resource file not found or not in path')
        super(WithinDocCoref, self).__init__()

    @staticmethod
    def arrange_resource(wd_mentions_json):
        document_tokens_dict = dict()
        for mention_json in wd_mentions_json:
            mention_data = MentionData.read_json_mention_data_line(mention_json)
            mention_tokens = mention_data.tokens_number
            for i in range(0, len(mention_tokens)):
                doc_id = mention_data.doc_id
                sent_id = mention_data.sent_id
                token_map_key = MentionData.static_gen_token_unique_id(doc_id, sent_id,
                                                                       mention_tokens[i])
                document_tokens_dict[token_map_key] = mention_data.coref_chain
        return document_tokens_dict

    def extract_all_relations(self, mention_x: MentionData,
                              mention_y: MentionData) -> Set[RelationType]:
        ret_ = set()
        ret_.add(self.extract_sub_relations(mention_x, mention_y, RelationType.WITHIN_DOC_COREF))
        return ret_

    def extract_sub_relations(self, mention_x: MentionData, mention_y: MentionData,
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
        if relation is not RelationType.WITHIN_DOC_COREF:
            return RelationType.NO_RELATION_FOUND

        if mention_x.doc_id == mention_y.doc_id:
            ment_x_coref_chain = self.extract_within_coref(mention_x)
            ment_y_coref_chain = self.extract_within_coref(mention_y)

            if not ment_x_coref_chain or not ment_y_coref_chain:
                return RelationType.NO_RELATION_FOUND

            if '-' in ment_x_coref_chain or '-' in ment_y_coref_chain:
                return RelationType.NO_RELATION_FOUND

            if set(ment_x_coref_chain) == set(ment_y_coref_chain):
                return RelationType.WITHIN_DOC_COREF

        return RelationType.NO_RELATION_FOUND

    def extract_within_coref(self, mention: MentionData) -> List[str]:
        tokens = mention.tokens_number
        within_coref_token = []
        for token_id in tokens:
            token_x_id = MentionData.static_gen_token_unique_id(str(mention.doc_id),
                                                                str(mention.sent_id),
                                                                str(token_id))
            if token_x_id in self.within_doc_coref_chain:
                token_coref_chain = self.within_doc_coref_chain[token_x_id]
                if token_coref_chain:
                    within_coref_token.append(token_coref_chain)
            else:
                within_coref_token.append('-')
                break

        return within_coref_token

    def get_within_doc_coref_chain(self):
        return self.within_doc_coref_chain

    @staticmethod
    def create_ment_id(mention_x: MentionData, mention_y: MentionData) -> str:
        return '_'.join([mention_x.get_mention_id(), mention_y.get_mention_id()])

    @staticmethod
    def get_supported_relations() -> List[RelationType]:
        """
        Return all supported relations by this class

        Returns:
            List[RelationType]
        """
        return [RelationType.WITHIN_DOC_COREF]
