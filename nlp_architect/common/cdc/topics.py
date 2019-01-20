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
from typing import List

from nlp_architect.common.cdc.mention_data import MentionData
from nlp_architect.utils.io import load_json_file

logger = logging.getLogger(__name__)


class Topic(object):
    def __init__(self, topic_id):
        self.topic_id = topic_id
        self.mentions = []


class Topics(object):
    def __init__(self):
        self.topics_list = []
        self.keep_order = False

    def create_from_file(self, mentions_file_path: str, keep_order: bool = False) -> None:
        """

        Args:
            keep_order: whether to keep original mentions order or not (default = False)
            mentions_file_path: this topic mentions json file
        """
        self.keep_order = keep_order
        self.topics_list = self.load_mentions_from_file(mentions_file_path)

    def load_mentions_from_file(self, mentions_file_path: str) -> List[Topic]:
        start_data_load = time.time()
        logger.info('Loading mentions from-%s', mentions_file_path)
        mentions = load_json_file(mentions_file_path)
        topics = self.order_mentions_by_topics(mentions)
        end_data_load = time.time()
        took_load = end_data_load - start_data_load
        logger.info('Mentions file-%s, took:%.4f sec to load', mentions_file_path, took_load)
        return topics

    def order_mentions_by_topics(self, mentions: str) -> List[Topic]:
        """
        Order mentions to documents topics
        Args:
            mentions: json mentions file

        Returns:
            List[Topic] of the mentions separated by their documents topics
        """
        running_index = 0
        topics = []
        current_topic_ref = None
        for mention_line in mentions:
            mention = MentionData.read_json_mention_data_line(mention_line)

            if self.keep_order:
                if mention.mention_index == -1:
                    mention.mention_index = running_index
                    running_index += 1

            topic_id = mention.topic_id

            if not current_topic_ref or len(topics) > 0 and topic_id != topics[-1].topic_id:
                current_topic_ref = Topic(topic_id)
                topics.append(current_topic_ref)

            current_topic_ref.mentions.append(mention)

        return topics
