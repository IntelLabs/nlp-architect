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


def simple_ensembler(np_arrays, weights):
    """
    Simple ensembler takes a list of n by m numpy array predictions and a weight list
    The predictions should be n by m. n is the number of elements and m is the number of classes


    Modified from the default LookupTable implementation to support multiple axis lookups.

    Args:
        vocab_size (int): the vocabulary size
        embed_dim (int): the size of embedding vector
        init (Initializor): initialization function
        update (bool): if the word vectors get updated through training
        pad_idx (int): by knowing the pad value, the update will make sure always
                       have the vector representing pad value to be 0s.
    """
    ensembled_matrix = np_arrays[0] * weights[0]
    for i in range(1, len(np_arrays)):
        ensembled_matrix = ensembled_matrix + np_arrays[i] * weights[i]
    return ensembled_matrix
