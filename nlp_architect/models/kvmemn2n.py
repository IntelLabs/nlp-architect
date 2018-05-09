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
import ngraph as ng
from ngraph.frontends.neon import Layer, LookupTable
from ngraph.frontends.neon.axis import shadow_axes_map
from ngraph.frontends.neon import Layer
from ngraph.frontends.neon import GaussianInit
from ngraph.frontends.neon.graph import SubGraph
from nlp_architect.utils.encodings import position_encoding
from nlp_architect.contrib.ngraph.modified_lookup_table import ModifiedLookupTable


class KVMemN2N(Layer):

    """
    Key Value Memory Network model

    This class is an N-Graph implementation of the paper https://arxiv.org/abs/1606.03126
    "Key-Value Memory Networks for Directly Reading Documents"

    Args:
        num_iterations (int): number of batches per epoch
        batch_size (int): number of samples in a batch, used to create axis
        emb_size (int): embedding size
        nhops (int): number of internal memory hops
        story_length (int): maximum number of memories associated with an entity
        memory_size (int): maximum length of memory statements
        vocab_size (int): number of objects in dictionary
        vocab_axis (axis): the vobaulary axis
        use_v_luts (bool): if true, a separate lookup table will be used for each memory hop

    Returns:
        a_pred (tensor, vocab x batch): probabilities of each potential response (vocab dictionary)
        a_logits (tensor, batch): predicted answer for each question in the batch

    """
    def __init__(self, num_iterations, batch_size, emb_size, nhops,
                 story_length, memory_size, vocab_size, vocab_axis, use_v_luts):

        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.nhops = nhops
        self.story_length = story_length
        self.memory_size = memory_size
        self.vocab_size = vocab_size
        self.use_v_luts = use_v_luts

        # Create graph
        # Make axes
        self.batch_axis = ng.make_axis(length=batch_size, name='N')
        self.sentence_axis = ng.make_axis(length=story_length, name='sentence_axis')
        self.sentence_rec_axis = ng.make_axis(length=story_length, name='REC')
        self.memory_axis = ng.make_axis(length=memory_size, name='memory_axis')

        self.val_len_axis = ng.make_axis(length=1, name='REC')

        self.embedding_axis = ng.make_axis(length=emb_size, name='F')

        self.vocab_axis = vocab_axis

        # weight initializationn
        self.init = GaussianInit(mean=0.0, std=0.1)
        # Create constant position encoding tensor to multiply elementwise with embedded words
        self.pos_enc = position_encoding(self.sentence_rec_axis, self.embedding_axis)

        # Weight sharing
        self.LUT_A = ModifiedLookupTable(self.vocab_size, self.emb_size, self.init, update=True,
                                         pad_idx=0, name='LUT_A')
        if use_v_luts:
            self.LUTs_C = [ModifiedLookupTable(self.vocab_size, self.emb_size, self.init,
                           update=True, pad_idx=0) for n in range(self.nhops)]

    def __call__(self, inputs):
        query = ng.cast_axes(inputs['query'], [self.batch_axis, self.sentence_rec_axis])

        # Query embedding [batch, sentence_axis, F]
        q_emb = self.LUT_A(query)

        # Multiply by position encoding and sum
        u_0 = ng.sum(q_emb * self.pos_enc, reduction_axes=[self.sentence_rec_axis])  # [batch, F]

        # Start a list of the internal states of the model.
        # Will be appended to after each memory hop
        u = [u_0]

        for hopn in range(self.nhops):
            keys = ng.cast_axes(inputs['keys'], [self.batch_axis, self.memory_axis,
                                self.sentence_rec_axis])
            value = ng.cast_axes(inputs['values'], [self.batch_axis, self.memory_axis,
                                 self.val_len_axis])

            # Embed keys
            m_emb_A = self.LUT_A(keys)
            m_A = ng.sum(m_emb_A * self.pos_enc,
                         reduction_axes=[self.sentence_rec_axis])  # [batch, memory_axis, F]

            # Compute scalar similarity between internal state and each memory
            # Equivalent to dot product between u[-1] and each memory in m_A
            dotted = ng.sum(u[-1] * m_A, reduction_axes=[self.embedding_axis])

            probs = ng.softmax(dotted, self.memory_axis)  # [batch, memory_axis]

            # Embed values with same embedding as keys, or new LUTs
            if self.use_v_luts:
                m_emb_C = self.LUTs_C[hopn](value)
            else:
                m_emb_C = self.LUT_A(value)

            m_C = ng.sum(m_emb_C * self.pos_enc, reduction_axes=[self.sentence_rec_axis])

            # Compute weighted sum of output embeddings
            o_k = ng.sum(probs * m_C, reduction_axes=[self.memory_axis])  # [batch, F]

            u_k = u[-1] + o_k  # [batch, F]

            # Add new internal state
            u.append(u_k)

        # Compute predicted answer from product of final internal state and final LUT weight matrix
        if self.use_v_luts:
            a_logits = ng.dot(self.LUTs_C[-1].W, u[-1])  # [batch, V]
        else:
            a_logits = ng.dot(self.LUT_A.W, u[-1])  # [batch, V]
        # rename V to vocab_axis to match answer
        a_logits = ng.cast_axes(a_logits, [self.vocab_axis, self.batch_axis])
        a_pred = ng.softmax(a_logits, self.vocab_axis)

        return a_pred, a_logits
