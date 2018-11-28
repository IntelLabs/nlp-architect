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
import re
from collections import Counter

from nlp_architect.utils.text import SpacyInstance


class MatchLSTMAnswerPointer(object):
    """
    Defines end to end MatchLSTM and Answer_Pointer network for Reading Comprehension
    """

    def __init__(self, params_dict, embeddings):
        """
        Args:
            params_dict: Dictionary containing the following keys-
                         'max_question' : max length of all questions in the dataset
                         'max_para' :  max length of all paragraphs in the dataset
                         'hidden_size': number of hidden units in the network
                         'batch_size' : batch size defined by user

            embeddings: Glove pretrained embedding matrix
        """

        # Assign Variables:
        self.max_question = params_dict['max_question']
        self.max_para = params_dict['max_para']
        self.hidden_size = params_dict['hidden_size']
        self.batch_size = params_dict['batch_size']
        self.embeddings = embeddings
        self.inference_only = params_dict['inference_only']
        self.G_i = None
        self.attn = None
        self.stacked_lists_forward = None
        self.stacked_lists_reverse = None
        self.logits_withsf = None

        # init tokenizer
        self.tokenizer = SpacyInstance(disable=['tagger', 'ner', 'parser', 'vectors', 'textcat'])

        # Create Placeholders
        # Question ids
        self.question_ids = tf.placeholder(tf.int32, shape=[None, self.max_question],
                                           name="question_ids")
        # Paragraph ids
        self.para_ids = tf.placeholder(tf.int32, shape=[None, self.max_para],
                                       name="para_ids")
        # Length of question
        self.question_length = tf.placeholder(tf.int32, shape=[None],
                                              name="question_len")
        # Length of paragraph
        self.para_length = tf.placeholder(tf.int32, shape=[None],
                                          name="para_len")
        # Mask for paragraph
        self.para_mask = tf.placeholder(tf.float32, shape=[None, self.max_para],
                                        name="para_mask")
        # Mask for question
        self.ques_mask = tf.placeholder(tf.float32, shape=[None, self.max_question],
                                        name="ques_mask")
        # Answer spans
        if self.inference_only is False:
            self.labels = tf.placeholder(tf.int32, shape=[None, 2], name="labels")
        # Dropout value
        self.dropout = tf.placeholder(tf.float32, shape=[], name="dropout")
        self.global_step = tf.Variable(0, name='global')

        # Get variables
        self.create_variables()

        # Define model
        self.create_model()

    def create_variables(self):
        """
        Function to create variables used for training

        """
        # define all variables required for training
        self.W_p = tf.get_variable("W_p", [1, self.hidden_size, self.hidden_size])
        self.W_r = tf.get_variable("W_r", [1, self.hidden_size, self.hidden_size])
        self.W_q = tf.get_variable("W_q", [1, self.hidden_size, self.hidden_size])
        self.w_lr = tf.get_variable("w_lr", [1, 1, self.hidden_size])
        self.b_p = tf.get_variable("b_p", [1, self.hidden_size, 1])
        self.c_p = tf.get_variable("c_p", [1])
        self.ones_vector = tf.constant(np.ones([1, self.max_question]), dtype=tf.float32)

        self.ones_vector_exp = tf.tile(tf.expand_dims(self.ones_vector, 0),
                                       [self.batch_size, 1, 1])

        self.ones_vector_para = tf.constant(np.ones([1, self.max_para]), dtype=tf.float32)

        self.ones_para_exp = tf.tile(tf.expand_dims(self.ones_vector_para, 0),
                                     [self.batch_size, 1, 1])

        self.ones_embed = tf.tile(tf.expand_dims(tf.constant(
                                  np.ones([1, self.hidden_size]), dtype=tf.float32), 0),
                                  [self.batch_size, 1, 1])

        self.V_r = tf.get_variable("V_r", [1, self.hidden_size, 2 * self.hidden_size])
        self.W_a = tf.get_variable("W_a", [1, self.hidden_size, self.hidden_size])
        self.b_a = tf.get_variable("b_a", [1, self.hidden_size, 1])
        self.v_a_pointer = tf.get_variable("v_a_pointer", [1, 1, self.hidden_size])
        self.c_pointer = tf.get_variable("c_pointer", [1, 1, 1])
        self.Wans_q = tf.get_variable("Wans_q", [1, self.hidden_size, self.hidden_size])
        self.Wans_v = tf.get_variable("Wans_v", [1, self.hidden_size, self.hidden_size])
        self.Vans_r = tf.get_variable("Vans_r", [1, self.hidden_size, self.max_question])

        self.mask_ques_mul = tf.matmul(tf.transpose(self.ones_embed, [0, 2, 1]),
                                       tf.expand_dims(self.ques_mask, 1))

    def create_model(self):
        """
        Function to set up the end 2 end reading comprehension model

        """
        # Embedding Layer
        embedding_lookup = tf.Variable(self.embeddings, name="word_embeddings",
                                       dtype=tf.float32, trainable=False)
        # Embedding Lookups
        self.question_emb = tf.nn.embedding_lookup(embedding_lookup, self.question_ids,
                                                   name="question_embed")

        self.para_emb = tf.nn.embedding_lookup(embedding_lookup, self.para_ids,
                                               name="para_embed")

        # Apply dropout after embeddings
        self.question = tf.nn.dropout(self.question_emb, self.dropout)
        self.para = tf.nn.dropout(self.para_emb, self.dropout)

        # Encoding Layer
        # Share weights of pre-processing LSTM layer with both para and
        # question
        with tf.variable_scope("encoded_question"):
            self.lstm_cell_question = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size,
                                                                   state_is_tuple=True)

            self.encoded_question, _ = tf.nn.dynamic_rnn(self.lstm_cell_question,
                                                         self.question,
                                                         self.question_length,
                                                         dtype=tf.float32)

        with tf.variable_scope("encoded_para"):
            self.encoded_para, _ = tf.nn.dynamic_rnn(self.lstm_cell_question, self.para,
                                                     self.para_length, dtype=tf.float32)

        # Define Match LSTM and Answer Pointer Cells
        with tf.variable_scope("match_lstm_cell"):
            self.match_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size,
                                                                state_is_tuple=True)

        with tf.variable_scope("answer_pointer_cell"):
            self.lstm_pointer_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size,
                                                                  state_is_tuple=True)

        print('Match LSTM Pass')
        # Match LSTM Pass in forward direction
        self.unroll_with_attention(reverse=False)
        self.encoded_para_reverse = tf.reverse(self.encoded_para, axis=[1])
        # Match LSTM Pass in reverse direction
        self.unroll_with_attention(reverse=True)
        # Apply dropout
        self.stacked_lists = tf.concat([tf.nn.dropout(self.stacked_lists_forward,
                                                      tf.maximum(self.dropout, 0.8)),
                                        tf.nn.dropout(self.stacked_lists_reverse,
                                                      tf.maximum(self.dropout, 0.8))], 1)

        # Answer pointer pass
        self.logits = self.answer_pointer_pass()
        if self.inference_only is False:
            print('Settting up Loss')
            # Compute Losses
            loss_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits[0],
                                                                    labels=self.labels[:, 0])

            loss_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits[1],
                                                                    labels=self.labels[:, 1])
            # Total Loss
            self.loss = tf.reduce_mean(loss_1 + loss_2)
            self.learning_rate = tf.constant(0.002)

            print('Set up optmizer')
            # Optmizer
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate). \
                minimize(self.loss, global_step=self.global_step)

    def unroll_with_attention(self, reverse=False):
        """
        Function to run the match_lstm pass in both forward and reverse directions

        Args:
        reverse: Boolean indicating whether to unroll in reverse directions

        """
        # Intitialze first hidden_state with zeros
        h_r_old = tf.constant(
            np.zeros([self.batch_size, self.hidden_size, 1]), dtype=tf.float32)
        final_state_list = []

        for i in range(self.max_para):

            if not reverse:
                encoded_paraslice = tf.gather(self.encoded_para, indices=i, axis=1)
            else:
                encoded_paraslice = tf.gather(self.encoded_para_reverse, indices=i,
                                              axis=1)

            W_p_expanded = tf.tile(self.W_p, [self.batch_size, 1, 1])
            W_q_expanded = tf.tile(self.W_q, [self.batch_size, 1, 1])
            W_r_expanded = tf.tile(self.W_r, [self.batch_size, 1, 1])
            w_lr_expanded = tf.tile(self.w_lr, [self.batch_size, 1, 1])
            b_p_expanded = tf.tile(self.b_p, [self.batch_size, 1, 1])

            int_sum = tf.matmul(W_p_expanded, tf.expand_dims(encoded_paraslice, 2)) + \
                tf.matmul(W_r_expanded, h_r_old) + b_p_expanded

            int_sum_new = tf.matmul(int_sum, tf.expand_dims(self.ques_mask, 1))

            int_sum1 = tf.matmul(W_q_expanded,
                                 tf.transpose(self.encoded_question, [0, 2, 1]))

            self.G_i = tf.nn.tanh(int_sum_new + int_sum1) + \
                tf.expand_dims(self.c_p * self.ques_mask, 1)

            # Attention Vector
            self.attn = tf.nn.softmax(tf.matmul(w_lr_expanded, self.G_i))

            z1 = encoded_paraslice

            z2 = tf.squeeze(tf.matmul(tf.transpose(self.encoded_question, [0, 2, 1]),
                                      tf.transpose(self.attn, [0, 2, 1])), axis=2)

            z_i_stacked = tf.concat([z1, z2], 1)
            if i == 0:
                h_r_old, cell_state_old = self.match_lstm_cell(
                    z_i_stacked, state=self.match_lstm_cell.zero_state(self.batch_size,
                                                                       dtype=tf.float32))
            else:
                h_r_old, cell_state_old = self.match_lstm_cell(z_i_stacked,
                                                               state=cell_state_old)

            final_state_list.append(h_r_old)
            h_r_old = tf.expand_dims(h_r_old, 2)
            stacked_lists = tf.stack(final_state_list, 1)

        if not reverse:
            # Mask Output
            mask_mult_lstm_forward = tf.matmul(tf.transpose(self.ones_embed, [0, 2, 1]),
                                               tf.expand_dims(self.para_mask, 1))

            self.stacked_lists_forward = tf.multiply(tf.transpose(stacked_lists, [0, 2, 1]),
                                                     mask_mult_lstm_forward)
        else:
            # Mask Output
            mask_mult_lstm_reverse = tf.matmul(tf.transpose(
                self.ones_embed, [0, 2, 1]), tf.expand_dims(tf.reverse(
                                                            self.para_mask, axis=[1]), 1))

            self.stacked_lists_reverse = tf.reverse(tf.multiply(tf.transpose(
                stacked_lists, [0, 2, 1]), mask_mult_lstm_reverse), axis=[2])

    def answer_pointer_pass(self):
        """
        Function to run the answer pointer pass:

        Args:
            None

        Returns:
            List of logits for start and end indices of the answer
        """

        V_r_expanded = tf.tile(self.V_r, [self.batch_size, 1, 1])
        W_a_expanded = tf.tile(self.W_a, [self.batch_size, 1, 1])
        b_a_expanded = tf.tile(self.b_a, [self.batch_size, 1, 1])
        mask_multiplier_1 = tf.expand_dims(self.para_mask, 1)
        mask_multiplier = self.ones_para_exp

        v_apointer_exp = tf.tile(self.v_a_pointer, [self.batch_size, 1, 1])

        # Zero initialization
        h_k_old = tf.constant(
            np.zeros([self.batch_size, self.hidden_size, 1]), dtype=tf.float32)

        b_k_lists = []

        print("Answer Pointer Pass")

        for i in range(0, 2):
            sum1 = tf.matmul(V_r_expanded, self.stacked_lists)
            sum2 = tf.matmul(W_a_expanded, h_k_old) + b_a_expanded
            F_k = tf.nn.tanh(sum1 + tf.matmul(sum2, mask_multiplier))

            b_k_withoutsf = tf.matmul(v_apointer_exp, F_k)

            b_k = tf.nn.softmax(b_k_withoutsf + tf.log(mask_multiplier_1))
            lstm_cell_inp = tf.squeeze(tf.matmul(self.stacked_lists,
                                       tf.transpose(b_k, [0, 2, 1])), axis=2)

            with tf.variable_scope("lstm_pointer"):
                if i == 0:
                    h_k_old, cell_state_pointer = self.lstm_pointer_cell(
                        lstm_cell_inp, state=self.lstm_pointer_cell.zero_state(self.batch_size,
                                                                               dtype=tf.float32))
                else:
                    h_k_old, cell_state_pointer = self.lstm_pointer_cell(lstm_cell_inp,
                                                                         state=cell_state_pointer)

            h_k_old = tf.expand_dims(h_k_old, 2)
            b_k_lists.append(b_k_withoutsf + tf.log(mask_multiplier_1))

        self.logits_withsf = [tf.nn.softmax(tf.squeeze(b_k_lists[0], axis=1)),
                              tf.nn.softmax(tf.squeeze(b_k_lists[1], axis=1))]

        return [tf.squeeze(b_k_lists[0], axis=1), tf.squeeze(b_k_lists[1], axis=1)]

    @staticmethod
    def obtain_indices(preds_start, preds_end):
        """
        Function to get answer indices given the predictions

        Args:
            preds_start: predicted start indices
            predictions: predicted end indices

        Returns:
            final start and end indices for the answer
        """
        ans_start = []
        ans_end = []
        for i in range(preds_start.shape[0]):
            max_ans_id = -100000000
            st_idx = 0
            en_idx = 0
            ele1 = preds_start[i]
            ele2 = preds_end[i]
            len_para = len(ele1)
            for j in range(len_para):
                for k in range(15):
                    if j + k >= len_para:
                        break
                    ans_start_int = ele1[j]
                    ans_end_int = ele2[j + k]
                    if (ans_start_int + ans_end_int) > max_ans_id:
                        max_ans_id = ans_start_int + ans_end_int
                        st_idx = j
                        en_idx = j + k

            ans_start.append(st_idx)
            ans_end.append(en_idx)

        return (np.array(ans_start), np.array(ans_end))

    def cal_f1_score(self, ground_truths, predictions):
        """
        Function to calculate F-1 and EM scores

        Args:
            ground_truths: labels given in the dataset
            predictions: logits predicted by the network

        Returns:
            F1 score and Exact-Match score
        """
        start_idx, end_idx = self.obtain_indices(predictions[0], predictions[1])
        f1 = 0
        exact_match = 0
        for i in range(self.batch_size):
            ele1 = start_idx[i]
            ele2 = end_idx[i]
            preds = np.linspace(ele1, ele2, abs(ele2 - ele1 + 1))
            length_gts = abs(ground_truths[i][1] - ground_truths[i][0] + 1)
            gts = np.linspace(ground_truths[i][0], ground_truths[i][1], length_gts)
            common = Counter(preds) & Counter(gts)
            num_same = sum(common.values())

            exact_match += int(np.array_equal(preds, gts))
            if num_same == 0:
                f1 += 0
            else:
                precision = 1.0 * num_same / len(preds)
                recall = 1.0 * num_same / len(gts)
                f1 += (2 * precision * recall) / (precision + recall)

        return 100 * (f1 / self.batch_size), 100 * (exact_match / self.batch_size)

    def get_dynamic_feed_params(self, question_str, vocab_reverse):
        """
        Function to get required feed_dict format for user entered questions.
        Used mainly in the demo mode.

        Args:
           question_str: question string
           vocab_reverse: vocab dictionary with words as keys and indices as values

        Returns:
           question_idx: list of indicies represnting the question padded to max length
           question_len: actual length of the question
           ques_mask: mask for question_idx

        """

        question_words = [word.replace("``", '"').replace("''", '"')
                          for word in self.tokenizer.tokenize(question_str)]

        question_ids = [vocab_reverse[ele] for ele in question_words]

        if len(question_ids) < self.max_question:
            pad_length = self.max_question - len(question_ids)
            question_idx = question_ids + [0] * pad_length
            question_len = len(question_ids)
            ques_mask = np.zeros([1, self.max_question])
            ques_mask[0, 0:question_len] = 1
            ques_mask = ques_mask.tolist()[0]

        return question_idx, question_len, ques_mask

    def run_loop(self, session, train, mode='train', dropout=0.6):
        """
        Function to run training/validation loop and display training loss, F1 & EM scores

        Args:
            session: tensorflow session
            train:   data dictionary for training/validation
            dropout: float value
            mode: 'train'/'val'
        """

        nbatches = int((len(train['para']) / self.batch_size))
        f1_score = 0
        em_score = 0
        for idx in range(nbatches):
            # Train for all batches
            start_batch = self.batch_size * idx
            end_batch = self.batch_size * (idx + 1)
            if end_batch > len(train['para']):
                break

            # Create feed dictionary
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
            # Training Phase
            if mode == 'train':
                _, train_loss, _, logits, labels = session.run(
                    [self.optimizer, self.loss, self.learning_rate, self.logits_withsf,
                     self.labels], feed_dict=feed_dict_qa)

                if (idx % 20 == 0):
                    print('iteration = {}, train loss = {}'.format(idx, train_loss))
                    f1_score, em_score = self.cal_f1_score(labels, logits)
                    print("F-1 and EM Scores are", f1_score, em_score)

                self.global_step.assign(self.global_step + 1)

            else:
                logits, labels = session.run([self.logits_withsf, self.labels],
                                             feed_dict=feed_dict_qa)

                f1_score_int, em_score_int = self.cal_f1_score(labels, logits)
                f1_score += f1_score_int
                em_score += em_score_int

        # Validation Phase
        if mode == 'val':
            print(
                "Validation F1 and EM scores are",
                f1_score / nbatches,
                em_score / nbatches)

    # pylint: disable=inconsistent-return-statements
    def inference_mode(self, session, valid, vocab_tuple, num_examples, dropout=1.0,
                       dynamic_question_mode=False, dynamic_usr_question="",
                       dynamic_question_index=0):
        """
          Function to run inference_mode for reading comprehension

          Args:
              session: tensorflow session
              valid: data dictionary for validation set
              vocab_tuple: a tuple containing voacab dictionaries in forward and reverse directions
              num_examples : specify the number of samples to run for inference
              dropout : Float value which is always 1.0 for inference
              dynamic_question_mode : boolean to enable whether or not accept
                                      questions from the user(used in the demo mode)

        """
        vocab_forward = vocab_tuple[0]
        vocab_reverse = vocab_tuple[1]

        for idx in range(num_examples):
            if dynamic_question_mode is True:
                idx = dynamic_question_index
                required_params = self.get_dynamic_feed_params(dynamic_usr_question, vocab_reverse)
                question_ids = required_params[0]
                question_length = required_params[1]
                ques_mask = required_params[2]
                test_paragraph = [vocab_forward[ele] for ele in valid[idx][0] if ele != 0]
                para_string = " ".join(map(str, test_paragraph))
            else:
                # Print Paragraph
                print("\n")
                print("Paragraph Number:", idx)
                test_paragraph = [vocab_forward[ele] for ele in valid[idx][0] if ele != 0]
                para_string = " ".join(map(str, test_paragraph))
                print(re.sub(r'\s([?.!,"](?:\s|$))', r'\1', para_string))

                # Print corresponding Question
                test_question = [vocab_forward[ele] for ele in valid[idx][1] if ele != 0]
                ques_string = " ".join(map(str, test_question))
                print("Question:", re.sub(r'\s([?.!"",])', r'\1', ques_string))
                question_ids = valid[idx][1]
                question_length = valid[idx][3]
                ques_mask = valid[idx][6]

            # Create a feed dictionary
            feed_dict_qa = {self.para_ids: np.expand_dims(valid[idx][0], 0),
                            self.question_ids: np.expand_dims(question_ids, 0),
                            self.para_length: np.expand_dims(valid[idx][2], 0),
                            self.question_length: np.expand_dims(question_length, 0),
                            self.para_mask: np.expand_dims(valid[idx][5], 0),
                            self.ques_mask: np.expand_dims(ques_mask, 0),
                            self.dropout: dropout}

            # Run session and obtain indices
            predictions = session.run([self.logits_withsf], feed_dict=feed_dict_qa)

            # Get the start and end indices of the answer
            start_idx, end_idx = self.obtain_indices(predictions[0][0], predictions[0][1])
            answer_ind = valid[idx][0][start_idx[0]:end_idx[0] + 1]

            # Print answer
            req_ans = [vocab_forward[ele] for ele in answer_ind if ele != 0]
            ans_string = " ".join(map(str, req_ans))
            answer = re.sub(r'\s([?.!",])', r'\1', ans_string)
            print("Answer:", answer)
            if dynamic_question_mode is True:
                return {"answer": answer}
