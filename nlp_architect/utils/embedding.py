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


def load_word_embeddings(file_path):
    """
    Loads a word embedding model text file into a word(str) to numpy vector dictionary

    Args:
        file_path (str): path to model file
        emb_size (int): embedding vectors size

    Returns:
        list: a dictionary of numpy.ndarray vectors
        int: detected word embedding vector size
    """
    with open(file_path) as fp:
        word_vectors = {}
        size = None
        for line in fp:
            line_fields = line.split()
            if len(line_fields) < 5:
                continue
            else:
                if line[0] == ' ':
                    word_vectors[' '] = np.asarray(line_fields, dtype='float32')
                else:
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
