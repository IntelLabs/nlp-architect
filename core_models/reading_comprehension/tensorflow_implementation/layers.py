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
import numpy as np
import tensorflow as tf
from collections import Counter
import os
from random import shuffle
from utils import *


class Match_LSTM_AnswerPointer(object):
    """
    Class that defines the match lstm and answer pointer blocks
    """

    def __init__(self, params_dict, embeddings):

        # Assign Variables:
        self.params_dict = params_dict
        self.max_question = params_dict['max_question']
        self.max_para = params_dict['max_para']
        self.hidden_size = params_dict['hidden_size']
        self.batch_size = params_dict['batch_size']
        self.embeddings = embeddings

        # Create Placeholders
        # Question ids
        self.question_ids = tf.placeholder(
            tf.int32,
            shape=[
                None,
                self.max_question],
            name="question_ids")
        # Paragraph ids
        self.para_ids = tf.placeholder(
            tf.int32, shape=[
                None, self.max_para], name="para_ids")
        # Length of question
        self.question_length = tf.placeholder(
            tf.int32, shape=[None], name="question_len")
        # Length of paragraph
        self.para_length = tf.placeholder(
            tf.int32, shape=[None], name="para_len")
        # Mask for paragraph
        self.para_mask = tf.placeholder(
            tf.float32, shape=[
                None, self.max_para], name="para_mask")
        # Mask for question
        self.ques_mask = tf.placeholder(
            tf.float32, shape=[
                None, self.max_question], name="ques_mask")
        # Answer spans
        self.labels = tf.placeholder(tf.int32, shape=[None, 2], name="labels")
        # Dropout value
        self.dropout = tf.placeholder(tf.float32, shape=[], name="dropout")
        self.global_step = tf.Variable(0, name='global')

        # Get variables
        self.create_variables()

        # Define model
        self.create_model()

    def create_variables(self):

        # define all variables required for training
        self.W_p = tf.get_variable(
            "W_p", [1, self.hidden_size, self.hidden_size])
        self.W_r = tf.get_variable(
            "W_r", [1, self.hidden_size, self.hidden_size])
        self.W_q = tf.get_variable(
            "W_q", [1, self.hidden_size, self.hidden_size])
        self.w_lr = tf.get_variable("w_lr", [1, 1, self.hidden_size])
        self.b_p = tf.get_variable("b_p", [1, self.hidden_size, 1])
        self.c_p = tf.get_variable("c_p", [1])
        self.ones_vector = tf.constant(
            np.ones([1, self.max_question]), dtype=tf.float32)
        self.ones_vector_exp = tf.tile(
            tf.expand_dims(
                self.ones_vector, 0), [
                self.batch_size, 1, 1])
        self.ones_vector_para = tf.constant(
            np.ones([1, self.max_para]), dtype=tf.float32)
        self.ones_para_exp = tf.tile(
            tf.expand_dims(
                self.ones_vector_para, 0), [
                self.batch_size, 1, 1])
        self.ones_embed = tf.tile(tf.expand_dims(tf.constant(
            np.ones([1, self.hidden_size]), dtype=tf.float32), 0), [self.batch_size, 1, 1])

        self.V_r = tf.get_variable(
            "V_r", [1, self.hidden_size, 2 * self.hidden_size])
        self.W_a = tf.get_variable(
            "W_a", [1, self.hidden_size, self.hidden_size])
        self.b_a = tf.get_variable("b_a", [1, self.hidden_size, 1])
        self.v_a_pointer = tf.get_variable(
            "v_a_pointer", [1, 1, self.hidden_size])
        self.c_pointer = tf.get_variable("c_pointer", [1, 1, 1])
        self.Wans_q = tf.get_variable(
            "Wans_q", [1, self.hidden_size, self.hidden_size])
        self.Wans_v = tf.get_variable(
            "Wans_v", [1, self.hidden_size, self.hidden_size])
        self.Vans_r = tf.get_variable(
            "Vans_r", [1, self.hidden_size, self.max_question])
        self.v_ans_lr = tf.get_variable("v_ans_lr", [1, 1, self.hidden_size])

        self.mask_ques_mul = tf.matmul(
            tf.transpose(
                self.ones_embed, [
                    0, 2, 1]), tf.expand_dims(
                self.ques_mask, 1))

    def create_model(self):
        """
        Function to set up the end 2 end reading comprehension model

        """
        # Embedding Layer
        embedding_lookup = tf.Variable(
            self.embeddings,
            name="word_embeddings",
            dtype=tf.float32,
            trainable=False)
        self.question_emb = tf.nn.embedding_lookup(
            embedding_lookup, self.question_ids, name="question_embed")
        self.para_emb = tf.nn.embedding_lookup(
            embedding_lookup, self.para_ids, name="para_embed")

        # Apply dropout after embeddings
        self.question = tf.nn.dropout(self.question_emb, self.dropout)
        self.para = tf.nn.dropout(self.para_emb, self.dropout)

        # Encoding Layer
        # Share weights of pre-processing LSTM layer with both para and
        # question
        with tf.variable_scope("encoded_question"):
            self.lstm_cell_question = tf.nn.rnn_cell.BasicLSTMCell(
                self.hidden_size, state_is_tuple=True)
            self.encoded_question, _ = tf.nn.dynamic_rnn(
                self.lstm_cell_question, self.question, self.question_length, dtype=tf.float32)  # (-1, Q, H)

        with tf.variable_scope("encoded_para"):
            self.encoded_para, _ = tf.nn.dynamic_rnn(
                self.lstm_cell_question, self.para, self.para_length, dtype=tf.float32)

        # Define Match LSTM and Answer Pointer Cells
        with tf.variable_scope("match_lstm_cell"):
            self.match_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                self.hidden_size, state_is_tuple=True)

        with tf.variable_scope("answer_pointer_cell"):
            self.lstm_pointer_cell = tf.nn.rnn_cell.BasicLSTMCell(
                self.hidden_size, state_is_tuple=True)

        print('Match LSTM Pass')
        # Match LSTM Pass in forward direction
        self.unroll_with_attention(reverse=False)
        self.encoded_para_reverse = tf.reverse(self.encoded_para, axis=[1])
        # Match LSTM Pass in reverse direction
        self.unroll_with_attention(reverse=True)
        # Apply dropout
        self.stacked_lists = tf.concat(
            [
                tf.nn.dropout(
                    self.stacked_lists_forward, tf.maximum(
                        self.dropout, 0.8)), tf.nn.dropout(
                    self.stacked_lists_reverse, tf.maximum(
                        self.dropout, 0.8))], 1)

        # Answer pointer pass
        self.logits = self.answer_pointer_pass()

        print('Set up Loss')
        # Compute Losses
        loss_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits[0], labels=self.labels[:, 0])
        loss_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits[1], labels=self.labels[:, 1])

        self.loss = tf.reduce_mean(loss_1 + loss_2)
        self.learning_rate = tf.constant(0.002)

        print('Set up optmizer')
        # Optmizer
        self.optimizer = tf.train.AdamOptimizer(
            self.learning_rate).minimize(
            self.loss, global_step=self.global_step)

    def unroll_with_attention(self, reverse=False):
        """
        Function to run the match_lstm pass for both forward and reverse directions

        Arguments:
        ----------
        reverse: Boolean indicating whether to unroll in reverse directions
        Return:
        --------
        None
        """
        # Intitialze first hidden_state with zeros
        h_r_old = tf.constant(
            np.zeros([self.batch_size, self.hidden_size, 1]), dtype=tf.float32)
        final_state_list = []

        for i in range(self.max_para):

            if not reverse:
                encoded_paraslice = tf.gather(
                    self.encoded_para, indices=i, axis=1)
            else:
                encoded_paraslice = tf.gather(
                    self.encoded_para_reverse, indices=i, axis=1)

            W_p_expanded = tf.tile(self.W_p, [self.batch_size, 1, 1])
            W_q_expanded = tf.tile(self.W_q, [self.batch_size, 1, 1])
            W_r_expanded = tf.tile(self.W_r, [self.batch_size, 1, 1])
            w_lr_expanded = tf.tile(self.w_lr, [self.batch_size, 1, 1])
            b_p_expanded = tf.tile(self.b_p, [self.batch_size, 1, 1])
            int_sum = tf.matmul(W_p_expanded, tf.expand_dims(
                encoded_paraslice, 2)) + tf.matmul(W_r_expanded, h_r_old) + b_p_expanded
            int_sum_new = tf.matmul(int_sum, tf.expand_dims(self.ques_mask, 1))

            int_sum1 = tf.matmul(
                W_q_expanded, tf.transpose(
                    self.encoded_question, [
                        0, 2, 1]))
            # change back to int_sum please
            self.G_i = tf.nn.tanh(int_sum_new + int_sum1) + \
                tf.expand_dims(self.c_p * self.ques_mask, 1)
            self.attn = tf.nn.softmax(tf.matmul(w_lr_expanded, self.G_i))
            z1 = encoded_paraslice
            z2 = tf.squeeze(
                tf.matmul(
                    tf.transpose(
                        self.encoded_question, [
                            0, 2, 1]), tf.transpose(
                        self.attn, [
                            0, 2, 1])), axis=2)

            z_i_stacked = tf.concat([z1, z2], 1)
            if i == 0:
                h_r_old, cell_state_old = self.match_lstm_cell(
                    z_i_stacked, state=self.match_lstm_cell.zero_state(
                        self.batch_size, dtype=tf.float32))
            else:
                h_r_old, cell_state_old = self.match_lstm_cell(
                    z_i_stacked, state=cell_state_old)

            final_state_list.append(h_r_old)
            h_r_old = tf.expand_dims(h_r_old, 2)
            stacked_lists = tf.stack(final_state_list, 1)

        if not reverse:
            mask_mult_lstm_forward = tf.matmul(
                tf.transpose(
                    self.ones_embed, [
                        0, 2, 1]), tf.expand_dims(
                    self.para_mask, 1))
            # tf.transpose(stacked_lists,[0,2,1])#tf.multiply(tf.transpose(stacked_lists,[0,2,1]),mask_mult_lstm)
            self.stacked_lists_forward = tf.multiply(tf.transpose(
                stacked_lists, [0, 2, 1]), mask_mult_lstm_forward)
        else:
            mask_mult_lstm_reverse = tf.matmul(
                tf.transpose(
                    self.ones_embed, [
                        0, 2, 1]), tf.expand_dims(
                    tf.reverse(
                        self.para_mask, axis=[1]), 1))
            self.stacked_lists_reverse = tf.reverse(
                tf.multiply(
                    tf.transpose(
                        stacked_lists, [
                            0, 2, 1]), mask_mult_lstm_reverse), axis=[2])

    def answer_pointer_pass(self):
        """
        Function to run the answer pointer pass:

        Arguments:
        ---------
        None

        Return:
        --------
        List of logits for start and end indices of the answer
        """

        V_r_expanded = tf.tile(self.V_r, [self.batch_size, 1, 1])
        W_a_expanded = tf.tile(self.W_a, [self.batch_size, 1, 1])
        b_a_expanded = tf.tile(self.b_a, [self.batch_size, 1, 1])
        c_pointer_exp = tf.tile(self.c_pointer, [self.batch_size, 1, 1])
        Wans_q_expanded = tf.tile(self.Wans_q, [self.batch_size, 1, 1])
        Wans_v_question = tf.tile(self.Wans_v, [self.batch_size, 1, 1])
        Vans_r_expanded = tf.tile(self.Vans_r, [self.batch_size, 1, 1])
        v_ans_lr_exp = tf.tile(self.v_ans_lr, [self.batch_size, 1, 1])

        mask_multiplier_1 = tf.expand_dims(self.para_mask, 1)
        # tf.expand_dims(self.ones_vector_para,1)
        mask_multiplier = self.ones_para_exp

        v_apointer_exp = tf.tile(self.v_a_pointer, [self.batch_size, 1, 1])

        # h_k_old
        # Zero initialization
        h_k_old = tf.constant(
            np.zeros([self.batch_size, self.hidden_size, 1]), dtype=tf.float32)

        b_k_lists = []
        b_k_list_withsf = []

        # Set dropout value
        if self.dropout == 1:
            dp_val = 1
        else:
            dp_val = 0.8

        print("Answer Pointer Pass")

        for i in range(0, 2):
            sum1 = tf.matmul(V_r_expanded, self.stacked_lists)
            sum2 = tf.matmul(W_a_expanded, h_k_old) + b_a_expanded
            F_k = tf.nn.tanh(sum1 + tf.matmul(sum2, mask_multiplier))
            # tf.matmul(c_pointer_exp,mask_multiplier)

            b_k_withoutsf = tf.matmul(v_apointer_exp, F_k)

            b_k = tf.nn.softmax(
                b_k_withoutsf +
                tf.log(mask_multiplier_1))
            lstm_cell_inp = tf.squeeze(
                tf.matmul(
                    self.stacked_lists, tf.transpose(
                        b_k, [
                            0, 2, 1])))

            with tf.variable_scope("lstm_pointer"):
                if i == 0:
                    h_k_old, cell_state_pointer = self.lstm_pointer_cell(
                        lstm_cell_inp, state=self.lstm_pointer_cell.zero_state(
                            self.batch_size, dtype=tf.float32))
                else:
                    h_k_old, cell_state_pointer = self.lstm_pointer_cell(
                        lstm_cell_inp, state=cell_state_pointer)

            h_k_old = tf.expand_dims(h_k_old, 2)
            b_k_lists.append(
                b_k_withoutsf +
                tf.log(mask_multiplier_1))

        mask_mul2_withlog = tf.tile(tf.log(mask_multiplier_1), [1, 2, 1])
        mask_mul2 = tf.tile(mask_multiplier_1, [1, 2, 1])

        self.logits_withsf = [
            tf.nn.softmax(
                tf.squeeze(
                    b_k_lists[0])), tf.nn.softmax(
                tf.squeeze(
                    b_k_lists[1]))]

        #self.results = tf.argmax(self.logits_withsf, axis=2)

        return [tf.squeeze(b_k_lists[0]), tf.squeeze(b_k_lists[1])]

    def run_loop(self, session, train, mode='train', dropout=0.6):
        """
        Function to run training/validation loop
        Arguments:
        ----------
        session: tensorflow session
        train:   data dictionary for training/validation
        dropout: float value
        mode: 'train'/'val'
        Return:
        --------
        None
        """

        nbatches = int((len(train['para']) / self.batch_size))
        f1_score = 0
        em_score = 0
        for idx in range(nbatches):
            # train for all batches
            start_batch = self.batch_size * idx
            end_batch = self.batch_size * (idx + 1)
            if end_batch > len(train['para']):
                break

            # create feed dictionary
            feed_dict_qa = {
                self.para_ids: np.asarray(train['para'][start_batch:end_batch]),
                self.question_ids: np.asarray(train['question'][start_batch:end_batch]),
                self.para_length: np.asarray(train['para_len'][start_batch:end_batch]),
                self.question_length: np.asarray(train['question_len'][start_batch:end_batch]),
                self.labels: np.asarray(train['answer'][start_batch:end_batch]),
                self.para_mask: np.asarray(train['para_mask'][start_batch:end_batch]),
                self.ques_mask: np.asarray(train['question_mask'][start_batch:end_batch]),
                self.dropout: dropout
            }
            # Training Pass
            if mode == 'train':
                _, train_loss, l_rate, logits, labels = session.run(
                    [self.optimizer, self.loss, self.learning_rate, self.logits_withsf, self.labels], feed_dict=feed_dict_qa)

                if (idx % 20 == 0):
                    print("Iteration No and loss is", idx, train_loss, l_rate)
                    f1_score, em_score = cal_f1_score(
                        self.batch_size, labels, logits)
                    print("F-1 Score and EM is", f1_score, em_score)

                self.global_step.assign(self.global_step + 1)

            else:
                logits, labels = session.run(
                    [self.logits_withsf, self.labels], feed_dict=feed_dict_qa)

                f1_score_int, em_score_int = cal_f1_score(
                    self.batch_size, labels, logits)
                f1_score += f1_score_int
                em_score += em_score_int
        # validation Phase
        if mode == 'val':
            print(
                "Validation F1score and EM score is",
                f1_score / nbatches,
                em_score / nbatches)
