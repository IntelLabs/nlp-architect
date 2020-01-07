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
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from six.moves import range


def zero_nil_slot(t):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    t = tf.convert_to_tensor(t, name="t")
    s = tf.shape(t)[1]
    z = tf.zeros(tf.stack([1, s]))
    return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])])


class MemN2N_Dialog(object):
    """End-To-End Memory Network."""
    def __init__(self,
                 batch_size,
                 vocab_size,
                 sentence_size,
                 memory_size,
                 embedding_size,
                 num_cands,
                 max_cand_len,
                 hops=3,
                 max_grad_norm=40.0,
                 nonlin=None,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-8),
                 session=tf.Session(),
                 name='MemN2N_Dialog'):
        """Creates an End-To-End Memory Network for Goal Oriented Dialog

        Args:
            cands: Encoded candidate answers

            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            memory_size: The max size of the memory. Since Tensorflow currently does not support
            jagged arrays all memories must be padded to this length. If padding is required, the
            extra memories should be

            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to
            `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to
            `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._max_cand_len = max_cand_len
        self._num_cands = num_cands
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._name = name

        self._build_inputs()
        self._build_vars()

        self._opt = optimizer

        # cross entropy
        logits = self._inference(self._stories, self._queries)  # (batch_size, vocab_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=tf.cast(self._answers,
                                                                               tf.float32),
                                                                name="cross_entropy")
        # loss op
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(cross_entropy_sum)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        # assign ops
        self.loss_op = cross_entropy_sum
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)
        self.saver = tf.train.Saver(max_to_keep=1)

    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._num_cands], name="answers")
        self._cands = tf.placeholder(tf.int32, [None, self._num_cands, self._max_cand_len],
                                     name="candidate_answers")

    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat(axis=0, values=[nil_word_slot, self._init([self._vocab_size - 1,
                                                                     self._embedding_size])])
            W = tf.concat(axis=0, values=[nil_word_slot, self._init([self._vocab_size - 1,
                                                                     self._embedding_size])])

            self.LUT_A = tf.Variable(A, name="LUT_A")
            self.LUT_W = tf.Variable(W, name="LUT_W")

            # Dont use projection for layerwise weight sharing
            self.R_proj = tf.Variable(self._init([self._embedding_size, self._embedding_size]),
                                      name="R_proj")

        self._nil_vars = set([self.LUT_A.name, self.LUT_W.name])

    def _inference(self, stories, queries):
        with tf.variable_scope(self._name):
            q_emb = tf.nn.embedding_lookup(self.LUT_A, queries)
            u_0 = tf.reduce_sum(q_emb, 1)
            u = [u_0]

            for _ in range(self._hops):
                m_emb_A = tf.nn.embedding_lookup(self.LUT_A, stories)
                m_A = tf.reduce_sum(m_emb_A, 2)

                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m_A * u_temp, 2)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])

                # Reuse A for the output memory encoding
                c_temp = tf.transpose(m_A, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                # Project hidden state, and add update
                u_k = tf.matmul(u[-1], self.R_proj) + o_k

                # nonlinearity
                if self._nonlin:
                    u_k = self._nonlin(u_k)

                u.append(u_k)

            cands_emb = tf.nn.embedding_lookup(self.LUT_W, self._cands)
            cands_emb_sum = tf.reduce_sum(cands_emb, 2)

            logits = tf.reshape(tf.matmul(tf.expand_dims(u_k, 1),
                                          tf.transpose(cands_emb_sum, [0, 2, 1])),
                                (-1, cands_emb_sum.shape[1]))

            return logits

    def batch_fit(self, stories, queries, answers, cands):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._stories: stories, self._queries: queries,
                     self._answers: answers, self._cands: cands}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

        return loss

    def predict(self, stories, queries, cands):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._cands: cands}
        return self._sess.run(self.predict_op, feed_dict=feed_dict)
