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
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from nlp_architect.utils.text import Vocabulary


def load_word_embeddings(file_path, vocab=None):
    """
    Loads a word embedding model text file into a word(str) to numpy vector dictionary

    Args:
        file_path (str): path to model file
        vocab (list of str): optional - vocabulary

    Returns:
        list: a dictionary of numpy.ndarray vectors
        int: detected word embedding vector size
    """
    with open(file_path, encoding='utf-8') as fp:
        word_vectors = {}
        size = None
        for line in fp:
            line_fields = line.split()
            if len(line_fields) < 5:
                continue
            else:
                if line[0] == ' ':
                    word_vectors[' '] = np.asarray(line_fields, dtype='float32')
                elif vocab is None or line_fields[0] in vocab:
                    word_vectors[line_fields[0]] = np.asarray(line_fields[1:], dtype='float32')
                    if size is None:
                        size = len(line_fields[1:])
    return word_vectors, size


def fill_embedding_mat(src_mat, src_lex, emb_lex, emb_size):
    """
    Creates a new matrix from given matrix of int words using the embedding
    model provided.

    Args:
        src_mat (numpy.ndarray): source matrix
        src_lex (dict): source matrix lexicon
        emb_lex (dict): embedding lexicon
        emb_size (int): embedding vector size
    """
    emb_mat = np.zeros((src_mat.shape[0], src_mat.shape[1], emb_size))
    for i, sen in enumerate(src_mat):
        for j, w in enumerate(sen):
            if w > 0:
                w_emb = emb_lex.get(str(src_lex.get(w)).lower())
                if w_emb is not None:
                    emb_mat[i][j] = w_emb
    return emb_mat


def get_embedding_matrix(embeddings: dict, vocab: Vocabulary) -> np.ndarray:
    """
    Generate a matrix of word embeddings given a vocabulary

    Args:
        embeddings (dict): a dictionary of embedding vectors
        vocab (Vocabulary): a Vocabulary

    Returns:
        a 2D numpy matrix of lexicon embeddings
    """
    emb_size = len(next(iter(embeddings.values())))
    mat = np.zeros((vocab.max, emb_size))
    for word, wid in vocab.vocab.items():
        vec = embeddings.get(word.lower(), None)
        if vec is not None:
            mat[wid] = vec
    return mat


# pylint: disable=not-context-manager
class ELMoEmbedderTFHUB(object):
    def __init__(self):
        self.g = tf.Graph()

        with self.g.as_default():
            text_input = tf.placeholder(dtype=tf.string)
            text_input_size = tf.placeholder(dtype=tf.int32)
            print('Loading Tensorflow hub ELMo model, '
                  'might take a while on first load (downloading from web)')
            self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
            self.inputs = {
                'tokens': text_input,
                'sequence_len': text_input_size
            }
            self.embedding = self.elmo(inputs=self.inputs,
                                       signature='tokens',
                                       as_dict=True)['elmo']

            sess = tf.Session(graph=self.g)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            self.s = sess

    def get_vector(self, tokens):
        vec = self.s.run(self.embedding,
                         feed_dict={self.inputs['tokens']: [tokens],
                                    self.inputs['sequence_len']: [len(tokens)]})
        return np.squeeze(vec, axis=0)
