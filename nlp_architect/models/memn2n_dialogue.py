#!/usr/bin/env python
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
"""
Example implementation of End-to-End Memory Network modified slightly from the
baseic End-to-End Memory network to reproduce results on the Facebook bAbI
goal-oriented dialog dataset.

Reference:
    "Learning End-to-End Goal Oriented Dialog"
    https://arxiv.org/abs/1605.07683.
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import ngraph as ng
from ngraph.frontends.neon import Layer, GaussianInit
from nlp_architect.contrib.ngraph.modified_lookup_table import ModifiedLookupTable


class MemN2N_Dialog(Layer):
    """
    End-to-End Memory Networks for Goal Oriented Dialogue

    After the model is initialized, it accepts a BABI_Dialog class formatted dataset 
    as input and returns a probability distribution over candidate answers. 

    Args:
        cands (np.array): Vectorized array of potential candidate answers, encoded
            as integers, as returned by BABI_Dialog class. Shape = [num_cands, max_cand_length]
        num_cands (int): Number of potential candidate answers. 
        max_cand_len (int): Maximum length of a candidate answer sentence in number of words. 
        memory_size (int): Maximum number of sentences to keep in memory at any given time.
        max_utt_len (int): Maximum length of any given sentence / user utterance 
        vocab_size (int): Number of unique words in the vocabulary + 2 (0 is reserved for 
            a padding symbol, and 1 is reserved for OOV)
        emb_size (int): Dimensionality of word embeddings to use 
        batch_size (int): Number of training examples per batch 
        use_match_type (bool, optional): Flag to use match-type features
        kb_ents_to_type (dict, optional): For use with match-type features, dictionary of 
            entities found in the dataset mapping to their associated match-type
        kb_ents_to_cand_idxs (dict, optional): For use with match-type features, dictionary
            mapping from each entity in the  knowledge base to the set of indicies in the
            candidate_answers array that contain that entity.
        match_type_idxs (dict, optional): For use with match-type features, dictionary 
            mapping from match-type to the associated fixed index of the candidate vector
            which indicated this match type.
        nhop (int, optional): Number of memory-hops to perform during fprop 
        eps (float, optional): Small epsilon used for numerical stability in 
            softmax renormalization
        init (Initalizer, optional): Initalizer object used to initialize lookup table
            and projection layer.
    """
    def __init__(
        self,
        cands,
        num_cands,
        max_cand_len,
        memory_size,
        max_utt_len,
        vocab_size,
        emb_size,
        batch_size,
        use_match_type=False,
        kb_ents_to_type=None,
        kb_ents_to_cand_idxs=None,
        match_type_idxs=None,
        nhops=3,
        eps=1e-6,
        init=GaussianInit(
            mean=0.0,
            std=0.1)):
        super(MemN2N_Dialog, self).__init__()

        self.cands = cands
        self.memory_size = memory_size
        self.max_utt_len = max_utt_len
        self.vocab_size = vocab_size
        self.num_cands = num_cands
        self.max_cand_len = max_cand_len
        self.batch_size = batch_size
        self.use_match_type = use_match_type
        self.kb_ents_to_type = kb_ents_to_type
        self.kb_ents_to_cand_idxs = kb_ents_to_cand_idxs
        self.match_type_idxs = match_type_idxs
        self.nhops = nhops
        self.eps = eps
        self.init = init

        # Make axes
        self.batch_axis = ng.make_axis(length=batch_size, name='N')
        self.sentence_rec_axis = ng.make_axis(length=max_utt_len, name='REC')
        self.memory_axis = ng.make_axis(length=memory_size, name='memory_axis')
        self.embedding_axis = ng.make_axis(length=emb_size, name='F')
        self.embedding_axis_proj = ng.make_axis(length=emb_size, name='F_proj')
        self.cand_axis = ng.make_axis(length=num_cands, name='cand_axis')
        self.cand_rec_axis = ng.make_axis(length=max_cand_len, name='REC')

        # Weight sharing of A's accross all hops input and output
        self.LUT_A = ModifiedLookupTable(
            vocab_size, emb_size, init, update=True, pad_idx=0)
        # Use lookuptable W to embed the candidate answers
        self.LUT_W = ModifiedLookupTable(
            vocab_size, emb_size, init, update=True, pad_idx=0)

        # Initialize projection matrix between internal model states
        self.R_proj = ng.variable(
            axes=[
                self.embedding_axis,
                self.embedding_axis_proj],
            initial_value=init)

        if not self.use_match_type:
            # Initialize constant matrix of all candidate answers
            self.cands_mat = ng.constant(
                self.cands, axes=[
                    self.cand_axis, self.cand_rec_axis])

    def __call__(self, inputs):
        query = ng.cast_axes(
            inputs['user_utt'], [
                self.batch_axis, self.sentence_rec_axis])

        # Query embedding [batch, sentence_axis, F]
        q_emb = self.LUT_A(query)

        # Multiply by position encoding and sum
        u_0 = ng.sum(q_emb, reduction_axes=[self.sentence_rec_axis])

        # Start a list of the internal states of the model. Will be appended to
        # after each memory hop
        u = [u_0]

        for hopn in range(self.nhops):
            story = ng.cast_axes(
                inputs['memory'], [
                    self.batch_axis, self.memory_axis, self.sentence_rec_axis])

            # Re-use the query embedding matrix to embed the memory sentences
            # [batch, memory_axis, sentence_axis, F]
            m_emb_A = self.LUT_A(story)
            m_A = ng.sum(
                m_emb_A, reduction_axes=[
                    self.sentence_rec_axis])  # [batch, memory_axis, F]

            # Compute scalar similarity between internal state and each memory
            # Equivalent to dot product between u[-1] and each memory in m_A
            # [batch, memory_axis]
            dotted = ng.sum(u[-1] * m_A, reduction_axes=[self.embedding_axis])

            # [batch, memory_axis]
            probs = ng.softmax(dotted, self.memory_axis)

            # Renormalize probabilites according to non-empty memories
            probs_masked = probs * inputs['memory_mask']
            renorm_sum = ng.sum(
                probs_masked, reduction_axes=[
                    self.memory_axis]) + self.eps
            probs_renorm = (probs_masked + self.eps) / renorm_sum

            # Compute weighted sum of memory embeddings
            o_k = ng.sum(
                probs_renorm * m_A,
                reduction_axes=[
                    self.memory_axis])  # [batch, F]

            # Add the output back into the internal state and project
            u_k = ng.cast_axes(ng.dot(self.R_proj, o_k), [
                               self.embedding_axis, self.batch_axis]) + u[-1]  # [batch, F_proj]

            # Add new internal state
            u.append(u_k)

        if self.use_match_type:
            # [batch_axis, cand_axis, cand_rec_axis, F]
            self.cands_mat = inputs['cands_mat']

        # Embed all candidate responses using LUT_W
        # [<batch_axis>, cand_axis, cand_rec_axis, F]
        cand_emb_W = self.LUT_W(self.cands_mat)
        # No position encoding added yet
        cands_mat_emb = ng.sum(
            cand_emb_W, reduction_axes=[
                self.cand_rec_axis])  # [<batch_axis>, cand_axis, F]

        # Compute predicted answer from product of final internal state
        # and embedded candidate answers
        # a_logits = ng.dot(cands_mat_emb, u[-1]) # [batch, cand_axis]
        # [batch, cand_axis]
        a_logits = ng.sum(u[-1] * cands_mat_emb,
                          reduction_axes=[self.embedding_axis])

        # rename V to vocab_axis to match answer
        a_logits = ng.cast_axes(a_logits, [self.batch_axis, self.cand_axis])
        a_pred = ng.softmax(a_logits, self.cand_axis)

        return a_pred, probs_renorm
