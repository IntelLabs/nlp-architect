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
import pickle

import numpy as np
# from allennlp.commands.elmo import ElmoEmbedder

from nlp_architect.common.cdc.mention_data import MentionDataLight
from nlp_architect.utils.embedding import ELMoEmbedderTFHUB

logger = logging.getLogger(__name__)


class ElmoEmbedding(object):
    def __init__(self):
        logger.info('Loading Elmo Embedding module')
        self.embeder = ELMoEmbedderTFHUB()
        self.cache = dict()
        # self.embeder = ElmoEmbedder(options, weigths)
        logger.info('Elmo Embedding module lead successfully')

    def get_feature_vector(self, mention: MentionDataLight):
        if mention.mention_context:
            sentence = mention.mention_context
        else:
            sentence = mention.tokens_str

        return self.apply_get_from_cache(sentence)

    def apply_get_from_cache(self, sentence):
        if sentence in self.cache:
            elmo_avg = self.cache[sentence]
        else:
            elmo_avg = self.get_elmo_avg(sentence.split())
            self.cache[sentence] = elmo_avg

        return elmo_avg

    def get_avrg_feature_vector(self, tokens_str):
        if tokens_str is not None:
            return self.apply_get_from_cache(tokens_str)
        return None

    def get_elmo_avg(self, sentence):
        sentence_embedding = self.embeder.get_vector(sentence)
        return np.mean(sentence_embedding, axis=0)
        # sentence_embeding = self.embeder.embed_sentence(sentence)
        # embed_avg_layer = np.zeros(sentence_embeding.shape[2], dtype=np.float64)
        # for embed_layer in sentence_embeding:
        #     embed_avg_sent = np.zeros(sentence_embeding.shape[2], dtype=np.float64)
        #     for token_vec in embed_layer:
        #         embed_avg_sent = np.add(embed_avg_sent, token_vec)
        #
        #     embed_avg_sent = np.true_divide(embed_avg_sent, sentence_embeding.shape[1])
        #     embed_avg_layer = np.add(embed_avg_layer, embed_avg_sent)
        # return np.true_divide(embed_avg_layer, sentence_embeding.shape[0])


class ElmoEmbeddingOffline(object):
    def __init__(self, dump_file):
        logger.info('Loading Elmo Offline Embedding module')
        with open(dump_file, 'rb') as out:
            self.embeder = pickle.load(out)
        logger.info('Elmo Offline Embedding module lead successfully')

    def get_feature_vector(self, mention: MentionDataLight):
        embed = None
        ment_str = mention.tokens_str
        if ment_str in self.embeder:
            embed = self.embeder[ment_str]

        return embed

    def get_avrg_feature_vector(self, tokens_str):
        embed = None
        if tokens_str in self.embeder:
            embed = self.embeder[tokens_str]

        return embed
