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


def one_hot(mat, num_classes):
    """
    Convert a 1D matrix of ints into one-hot encoded vectors.

    Arguments:
        mat (numpy.ndarray): A 1D matrix of labels (int)
        num_classes (int): Number of all possible classes

    Returns:
        numpy.ndarray: A 2D matrix
    """
    assert len(mat.shape) < 2 or isinstance(mat.shape, int)
    vec = np.zeros((mat.shape[0], num_classes))
    for i, v in enumerate(mat):
        vec[i][v] = 1.0
    return vec


def one_hot_sentence(mat, num_classes):
    """
    Convert a 2D matrix of ints into one-hot encoded 3D matrix

    Arguments:
        mat (numpy.ndarray): A 2D matrix of labels (int)
        num_classes (int): Number of all possible classes

    Returns:
        numpy.ndarray: A 3D matrix
    """
    new_mat = []
    for i in range(mat.shape[0]):
        new_mat.append(one_hot(mat[i], num_classes))
    return np.asarray(new_mat)


def add_offset(mat, offset=1):
    """
    Add +1 to all values in matrix mat

    Arguments:
        mat (numpy.ndarray): A 2D matrix with int values
        offset (int): offset to add

    Returns:
        numpy.ndarray: input matrix
    """
    for i, vec in enumerate(mat):
        offset_arr = np.array(vec.shape)
        offset_arr.fill(offset)
        mat[i] = vec + offset_arr
    return mat
