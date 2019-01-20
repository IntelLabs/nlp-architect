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
import sys
from typing import List

from nlp_architect.utils.io import load_json_file
from nlp_architect.utils.string_utils import StringUtils


class MentionDataLight(object):
    def __init__(self, tokens_str: str, mention_context: str = None, mention_head: str = None,
                 mention_head_lemma: str = None, mention_pos: str = None,
                 mention_ner: str = None):
        """
        Object represent a mention with only text values
        Args:
            tokens_str: str the tokens combine text (join with space)
            mention_head: str
            mention_head_lemma: str
        """
        self.tokens_str = tokens_str
        self.mention_context = mention_context
        if not mention_head and not mention_head_lemma:
            self.mention_head, self.mention_head_lemma, self.mention_head_pos, \
                self.mention_ner = StringUtils.find_head_lemma_pos_ner(str(tokens_str))
        else:
            self.mention_head = mention_head
            self.mention_head_lemma = mention_head_lemma
            self.mention_head_pos = mention_pos
            self.mention_ner = mention_ner


class MentionData(MentionDataLight):
    def __init__(self, topic_id: str, doc_id: str, sent_id: int, tokens_numbers: List[int],
                 tokens_str: str, mention_context: List[str], mention_head: str,
                 mention_head_lemma: str, coref_chain: str, mention_type: str = 'NA',
                 is_continuous: bool = True, is_singleton: bool = False, score: float = float(-1),
                 predicted_coref_chain: str = None, mention_pos: str = None,
                 mention_ner: str = None, mention_index: int = -1) -> None:
        """
        Object represent a mention

        Args:
            topic_id: str topic ID
            doc_id: str document ID
            sent_id: int sentence number
            tokens_numbers: List[int] - tokens numbers
            mention_context: List[str] - list of tokens strings
            coref_chain: str
            mention_type: str one of (HUM/NON/TIM/LOC/ACT/NEG)
            is_continuous: bool
            is_singleton: bool
            score: float
            predicted_coref_chain: str (should be field while evaluated)
            mention_pos: str
            mention_ner: str
            mention_index: in case order is of value (default = -1)
        """
        super(MentionData, self).__init__(tokens_str, mention_context, mention_head,
                                          mention_head_lemma, mention_pos,
                                          mention_ner)
        self.topic_id = topic_id
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.tokens_number = tokens_numbers
        self.mention_type = mention_type
        self.coref_chain = coref_chain
        self.is_continuous = is_continuous
        self.is_singleton = is_singleton
        self.score = score
        self.predicted_coref_chain = predicted_coref_chain
        self.mention_id = self.gen_mention_id()
        self.mention_index = mention_index

    @staticmethod
    def read_json_mention_data_line(mention_line: str):
        """
        Args:
            mention_line: a Json representation of a single mention

        Returns:
            MentionData object
        """
        # pylint: disable=too-many-branches

        try:
            topic_id = None
            coref_chain = None
            doc_id = None
            sent_id = None
            tokens_numbers = None
            score = -1
            mention_type = None
            predicted_coref_chain = None
            mention_context = None
            is_continue = False
            is_singleton = False
            mention_pos = None
            mention_ner = None
            mention_index = -1

            mention_text = mention_line['tokens_str']

            if 'topic_id' in mention_line:
                topic_id = mention_line['topic_id']

            if 'coref_chain' in mention_line:
                coref_chain = mention_line['coref_chain']

            if 'doc_id' in mention_line:
                doc_id = mention_line['doc_id']
                if '.xml' not in doc_id:
                    doc_id = doc_id + '.xml'

            if 'sent_id' in mention_line:
                sent_id = mention_line['sent_id']

            if 'tokens_number' in mention_line:
                tokens_numbers = mention_line['tokens_number']

            if 'mention_context' in mention_line:
                mention_context = mention_line['mention_context']

            if 'mention_head' in mention_line and 'mention_head_lemma' in mention_line:
                mention_head = mention_line['mention_head']
                mention_head_lemma = mention_line['mention_head_lemma']
                if 'mention_head_pos' in mention_line:
                    mention_pos = mention_line['mention_head_pos']
                if 'mention_ner' in mention_line:
                    mention_ner = mention_line['mention_ner']
            else:
                mention_head, mention_head_lemma, mention_pos, \
                    mention_ner = StringUtils.find_head_lemma_pos_ner(str(mention_text))

            if 'mention_type' in mention_line:
                mention_type = mention_line['mention_type']
            if 'score' in mention_line:
                score = mention_line['score']

            if 'is_continuous' in mention_line:
                is_continue = mention_line['is_continuous']

            if 'is_singleton' in mention_line:
                is_singleton = mention_line['is_singleton']

            if 'predicted_coref_chain' in mention_line:
                predicted_coref_chain = mention_line['predicted_coref_chain']

            if 'mention_index' in mention_line:
                mention_index = mention_line['mention_index']

            mention_data = MentionData(topic_id, doc_id, sent_id, tokens_numbers, mention_text,
                                       mention_context,
                                       mention_head, mention_head_lemma,
                                       coref_chain, mention_type, is_continue, is_singleton, score,
                                       predicted_coref_chain, mention_pos, mention_ner,
                                       mention_index)
        except Exception:
            print('Unexpected error:', sys.exc_info()[0])
            raise Exception('failed reading json line-' + str(mention_line))

        return mention_data

    @staticmethod
    def read_mentions_json_to_mentions_data_list(mentions_json_file: str):
        """

        Args:
            mentions_json_file: the path of the mentions json file to read

        Returns:
            List[MentionData]
        """
        all_mentions_only = load_json_file(mentions_json_file)

        mentions = []
        for mention_line in all_mentions_only:
            mention_data = MentionData.read_json_mention_data_line(mention_line)
            mentions.append(mention_data)

        return mentions

    def get_tokens(self):
        return self.tokens_number

    def gen_mention_id(self) -> str:
        if self.doc_id and self.sent_id is not None and self.tokens_number:
            tokens_ids = [str(self.doc_id), str(self.sent_id)]
            tokens_ids.extend([str(token_id) for token_id in self.tokens_number])
            return '_'.join(tokens_ids)

        return '_'.join(self.tokens_str.split())

    def get_mention_id(self) -> str:
        if not self.mention_id:
            self.mention_id = self.gen_mention_id()
        return self.mention_id

    @staticmethod
    def static_gen_token_unique_id(doc_id: int, sent_id: int, token_id: int) -> str:
        return '_'.join([str(doc_id), str(sent_id), str(token_id)])
