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

from nlp_architect.common.cdc.mention_data import MentionDataLight

logger = logging.getLogger(__name__)


class GloveEmbedding(object):
    def __init__(self, glove_file):
        logger.info('Loading Glove Online Embedding module, This my take a while...')
        self.word_to_ix, self.word_embeddings = self.load_glove_for_vocab(glove_file)
        logger.info('Glove Offline Embedding module lead successfully')

    @staticmethod
    def load_glove_for_vocab(glove_filename):
        vocab = []
        embd = []
        with open(glove_filename) as glove_file:
            for line in glove_file:
                row = line.strip().split(' ')
                word = row[0]
                vocab.append(word)
                embd.append(row[1:])

        embeddings = np.asarray(embd, dtype=float)
        word_to_ix = {word: i for i, word in enumerate(vocab)}
        return word_to_ix, embeddings


class GloveEmbeddingOffline(object):
    def __init__(self, embed_resources):
        logger.info('Loading Glove Offline Embedding module')
        with open(embed_resources, 'rb') as out:
            self.word_to_ix, self.word_embeddings = pickle.load(out, encoding='latin1')
        logger.info('Glove Offline Embedding module lead successfully')

    def get_feature_vector(self, mention: MentionDataLight):
        embed = None
        head = mention.mention_head
        lemma = mention.mention_head_lemma
        if head in self.word_to_ix:
            embed = self.word_embeddings[self.word_to_ix[head]]
        elif lemma in self.word_to_ix:
            embed = self.word_embeddings[self.word_to_ix[lemma]]

        return embed

    def get_avrg_feature_vector(self, tokens_str):
        embed = np.zeros(300, dtype=np.float64)
        mention_size = 0
        for token in tokens_str.split():
            if token in self.word_to_ix:
                token_embed = self.word_embeddings[self.word_to_ix[token]]
                embed = np.add(embed, token_embed)
                mention_size += 1

        if mention_size == 0:
            mention_size = 1

        return np.true_divide(embed, mention_size)
