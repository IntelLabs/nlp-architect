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
from typing import List

from nlp_architect.common.cdc.cluster import Clusters
from nlp_architect.common.cdc.mention_data import MentionData
from nlp_architect.common.cdc.topics import Topic
from nlp_architect.utils.string_utils import StringUtils

logger = logging.getLogger(__name__)


def write_clusters_to_file(clusters: Clusters, topic_id: str, file_obj) -> None:
    """
    Write the clusters to a text file (for experiments or evaluation using
        coreference scorer (v8.01))
    Args:
        clusters: the cluster to write
        topic_id:
        file_obj: file object
    """
    i = 0
    file_obj.write('Topic - ' + topic_id + '\n')
    for cluster in clusters.clusters_list:
        i += 1
        file_obj.write('cluster #' + str(i) + '\n')
        mentions_list = []
        for mention in cluster.mentions:
            mentions_list.append((mention.tokens_str, mention.predicted_coref_chain))
        file_obj.write(str(mentions_list) + '\n')


def extract_vocab(mentions: List[MentionData], filter_stop_words: bool) -> List[str]:
    """
    Extract Head, Lemma and mention string from all mentions to create a list of string vocabulary
    Args:
        mentions:
        filter_stop_words:

    Returns:

    """
    vocab = set()
    for mention in mentions:
        head = mention.mention_head
        head_lemma = mention.mention_head_lemma
        tokens_str = mention.tokens_str
        if not filter_stop_words:
            vocab.add(head)
            vocab.add(head_lemma)
            vocab.add(tokens_str)
        else:
            if not StringUtils.is_stop(head):
                vocab.add(head)
            if not StringUtils.is_stop(head_lemma):
                vocab.add(head_lemma)
            if not StringUtils.is_stop(tokens_str):
                vocab.add(tokens_str)
    vocab_set = list(vocab)
    return vocab_set


def load_mentions_vocab_from_files(mentions_files, filter_stop_words=False):
    logger.info('Loading mentions files...')
    mentions = []
    for _file in mentions_files:
        mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_file))

    return load_mentions_vocab(mentions, filter_stop_words)


def load_mentions_vocab(mentions, filter_stop_words=False):
    vocab = extract_vocab(mentions, filter_stop_words)
    logger.info('Done loading mentions files...')
    return vocab


def write_event_coref_scorer_results(topics_list: List[Topic], output_file: str) -> None:
    with open(os.path.join(output_file, 'cd_event_pred_clusters_spans.txt'), 'w') as output:
        write_topics(topics_list, output)


def write_entity_coref_scorer_results(topics_list: List[Topic], output_file: str) -> None:
    with open(os.path.join(output_file, 'cd_entity_pred_clusters_spans.txt'), 'w') as output:
        write_topics(topics_list, output)


def write_topics(topics_list: List[Topic], output) -> None:
    output.write('#begin document (ECB+/ecbplus_all); part 000\n')
    for topic in topics_list:
        for mention in topic.mentions:
            output.write('ECB+/ecbplus_all\t' + '(' + str(mention.predicted_coref_chain) + ')\n')
    output.write('#end document')
