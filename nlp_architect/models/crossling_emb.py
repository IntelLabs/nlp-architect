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
from __future__ import print_function, division

import io
import os
import time

import numpy as np
import scipy
import tensorflow as tf


class Discriminator:

    def __init__(self, input_data, Y, lr_ph):
        self.input_data = input_data
        self.lr_ph = lr_ph
        self.do_ph = tf.placeholder(name="dropout_ph", dtype=tf.float32)
        self.Y = Y
        self.hid_dim = 2048
        # Build Graph
        self._build_network_graph()
        self.disc_cost = None
        self.disc_opt = None
        self.map_opt = None
        self.W = None

    def _build_network_graph(self):
        """
        Builds the basic inference graph for discriminator
        """
        with tf.variable_scope("Discriminator", reuse=tf.AUTO_REUSE):
            w_init = tf.contrib.layers.xavier_initializer()
            noisy_input = tf.nn.dropout(self.input_data, self.do_ph, name="DO1")
            fc1 = tf.layers.dense(noisy_input, self.hid_dim, kernel_initializer=w_init,
                                  activation=tf.nn.leaky_relu, name="Dense1")
            fc2 = tf.layers.dense(fc1, self.hid_dim, kernel_initializer=w_init,
                                  activation=tf.nn.leaky_relu, name="Dense2")
            self.prediction = tf.layers.dense(fc2, 1, kernel_initializer=w_init,
                                              name="Dense_Sig")

    def build_train_graph(self, disc_pred):
        """
        Builds training graph for discriminator
        Arguments:
             disc_pred(object): Discriminator instance
        """
        # Variables in discrimnator scope
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Discriminator")
        # Binary Cross entropy
        disc_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_pred, labels=self.Y)
        # Cost
        self.disc_cost = tf.reduce_mean(disc_entropy)
        # Optimizer
        disc_opt = tf.train.GradientDescentOptimizer(self.lr_ph)
        self.disc_opt = disc_opt.minimize(self.disc_cost, var_list=disc_vars)


class Generator:

    def __init__(self, src_ten, tgt_ten, emb_dim, batch_size, smooth_val, lr_ph, beta, vocab_size):
        self.src_ten = src_ten
        self.tgt_ten = tgt_ten
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.smooth_val = smooth_val
        self.beta = beta
        self.lr_ph = lr_ph
        self.vocab_size = vocab_size

        # Placeholders
        self.src_ph = tf.placeholder(name="src_ph", shape=[None], dtype=tf.int32)
        self.tgt_ph = tf.placeholder(name="tgt_ph", shape=[None], dtype=tf.int32)

        # Build Graph
        self._build_network_graph()
        ortho_weight = self._build_ortho_graph(self.W)
        self.assign_weight = self._assign_ortho_weight(ortho_weight)
        self.map_opt = None
        self.W = None

    def _build_network_graph(self):
        """
        Builds basic inference graph for generator
        """
        with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE):
            # Look up tables
            self.src_emb = tf.nn.embedding_lookup(self.src_ten, self.src_ph, name="src_lut")
            self.tgt_emb = tf.nn.embedding_lookup(self.tgt_ten, self.tgt_ph, name="tgt_lut")
            # Map them
            self.mapWX = self._mapper(self.src_emb)
            # Concatenate them
            self.X = tf.concat([self.mapWX, self.tgt_emb], 0, name="X")
            # Set target for discriminator
            Y = np.zeros(shape=(2 * self.batch_size, 1), dtype=np.float32)
            # Label smoothing
            Y[:self.batch_size] = 1 - self.smooth_val
            Y[self.batch_size:] = self.smooth_val
            # Convert to tensor
            self.Y = tf.convert_to_tensor(Y, name="Y")

    def build_train_graph(self, disc_pred):
        """
        Builds training graph for generator
        Arguments:
            disc_pred(object): Discriminator instance

        """
        # Variables in Mapper scope
        map_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Generator/Mapper")
        # Binary Cross entropy
        map_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_pred, labels=(1 - self.Y))
        # Cost
        map_cost = tf.reduce_mean(map_entropy)
        map_opt = tf.train.GradientDescentOptimizer(self.lr_ph)
        self.map_opt = map_opt.minimize(map_cost, var_list=map_vars)

    def _build_ortho_graph(self, W):
        """
        Builds a graph to orthogonalize weight W
        Arguments:
            W (Tensor): Weight in the mapper
        """
        with tf.variable_scope("Ortho", reuse=tf.AUTO_REUSE):
            a = tf.scalar_mul((1 + self.beta), W)  # (1+B)W
            b = tf.matmul(tf.transpose(W), W)  # WWt
            c = tf.matmul(W, b)  # W(W.Wt)
            d = tf.scalar_mul(self.beta, c)  # B(W.Wt)W
            ortho_weight = a - d
            return ortho_weight

    def _assign_ortho_weight(self, ortho_weight):
        """
        Builds a graph to assign weight W after it is orthogonalized
        Arguments:
             ortho_weight(Tensor): Weight after it is orthogonalized
        """
        return tf.assign(self.W, ortho_weight)

    def _mapper(self, src_emb):
        """
        Learns WX mapping to make ||WX-Y|| smaller
        Arguments:
             src_emb(Tensor): Source embeddings after lookup
        """
        with tf.variable_scope("Mapper", reuse=tf.AUTO_REUSE):
            # Initialize as an eye of emb_dim x emb_dim
            self.W = tf.Variable(name="W", initial_value=tf.eye(self.emb_dim, self.emb_dim))
            # Do Matrix Multiply
            WX = tf.matmul(src_emb, self.W)
            # Returns map and weight handles
            return WX


class WordTranslator:
    """
    Main network which does cross-lingual embeddings training
    """

    def __init__(self, hparams, src_vec, tgt_vec, vocab_size):
        # Hyperparameters
        self.batch_size = hparams.batch_size
        self.smooth_val = hparams.smooth_val
        self.beta = hparams.beta
        self.most_freq = hparams.most_freq
        self.emb_dim = hparams.emb_dim
        self.vocab_size = vocab_size
        self.disc_runs = hparams.disc_runs
        self.iters_epoch = hparams.iters_epoch
        self.src_vec = src_vec
        self.tgt_vec = tgt_vec
        self.src_ten = tf.convert_to_tensor(src_vec)
        self.tgt_ten = tf.convert_to_tensor(tgt_vec)
        self.save_dir = hparams.weight_dir
        self.slang = hparams.src_lang
        self.tlang = hparams.tgt_lang

        # Placeholders
        self.lr_ph = tf.placeholder(tf.float32, name="lrPh")
        # Build Graph
        self._build_network_graph()
        self._build_train_graph()

    def _build_network_graph(self):
        """
        Builds inference graph for the GAN
        """
        self.generator = Generator(self.src_ten, self.tgt_ten, self.emb_dim, self.batch_size,
                                   self.smooth_val, self.lr_ph, self.beta, self.vocab_size)
        self.discriminator = Discriminator(self.generator.X, self.generator.Y, self.lr_ph)

    def _build_train_graph(self):
        """
        Builds training graph for the GAN
        """
        self.generator.build_train_graph(self.discriminator.prediction)
        self.discriminator.build_train_graph(self.discriminator.prediction)

    @staticmethod
    def report_metrics(iters, n_words_proc, disc_cost_acc, tic):
        """
        Reports metrics of how training is going
        """
        if iters > 0 and iters % 500 == 0:
            mean_cost = str(sum(disc_cost_acc) / len(disc_cost_acc))
            print(str(int(n_words_proc / (time.time() - tic))) + " Samples/Sec - Iter "
                  + str(iters) + " Discriminator Cost: " + mean_cost)
            # Reset instrumentation
            del disc_cost_acc
            disc_cost_acc = []
            n_words_proc = 0
            tic = time.time()

    def run_generator(self, sess, local_lr):
        """
        Runs generator part of GAN
        Arguments:
            sess(tf.session): Tensorflow Session
            local_lr(float): Learning rate
        Returns:
            Returns number of words processed
        """
        # Generate random ids to look up
        src_ids = np.random.choice(self.vocab_size, self.batch_size, replace=False)
        tgt_ids = np.random.choice(self.vocab_size, self.batch_size, replace=False)
        train_dict = {self.generator.src_ph: src_ids,
                      self.generator.tgt_ph: tgt_ids,
                      self.discriminator.do_ph: 1.0,
                      self.lr_ph: local_lr}
        sess.run(self.generator.map_opt, feed_dict=train_dict)
        # Run orthogonalize
        sess.run(self.generator.assign_weight)
        return 2 * self.batch_size

    def run_discriminator(self, sess, local_lr):
        """
        Runs discriminator part of GAN
        Arguments:
            sess(tf.session): Tensorflow Session
            local_lr(float): Learning rate
        """
        # Generate random ids to look up
        src_ids = np.random.choice(self.most_freq, self.batch_size, replace=False)
        tgt_ids = np.random.choice(self.most_freq, self.batch_size, replace=False)
        train_dict = {self.generator.src_ph: src_ids,
                      self.generator.tgt_ph: tgt_ids,
                      self.discriminator.do_ph: 0.9,
                      self.lr_ph: local_lr}
        return sess.run([self.discriminator.disc_cost, self.discriminator.disc_opt],
                        feed_dict=train_dict)

    def run(self, sess, local_lr):
        """
        Runs whole GAN
        Arguments:
            sess(tf.session): Tensorflow Session
            local_lr(float): Learning rate
        """
        disc_cost_acc = []
        n_words_proc = 0
        tic = time.time()
        for iters in range(0, self.iters_epoch, self.batch_size):
            # 1.Run the discriminator
            for _ in range(self.disc_runs):
                disc_result = self.run_discriminator(sess, local_lr)
                disc_cost_acc.append(disc_result[0])
            # 2.Run the Generator
            n_words_proc += self.run_generator(sess, local_lr)
            # 3.Report the metrics
            self.report_metrics(iters, n_words_proc, disc_cost_acc, tic)

    @staticmethod
    def set_lr(local_lr, drop_lr):
        """
        Drops learning rate based on CSLS criterion
        Arguments:
            local_lr(float): Learning Rate
            drop_lr(bool): Drop learning rate by 2 if True
        """
        new_lr = local_lr * 0.98
        print("Dropping learning rate to " + str(new_lr) + " from " + str(local_lr))
        if drop_lr:
            new_lr = new_lr / 2.0
            print("Dividing learning rate by 2 as validation criterion\
                   decreased. New lr is " + str(new_lr))
        return new_lr

    def save_model(self, save_model, sess):
        """
        Saves W in mapper as numpy array based on CSLS criterion
        Arguments:
            save_model(bool): Save model if True
            sess(tf.session): Tensorflow Session
        """
        if save_model:
            print("Saving model ....")
            model_W = sess.run(self.generator.W)
            path = os.path.join(self.save_dir, "W_best_mapping")
            np.save(path, model_W)

    def apply_procrustes(self, sess, final_pairs):
        """
        Applies procrustes to W matrix for better mapping
        Arguments:
            sess(tf.session): Tensorflow Session
            final_pairs(ndarray): Array of pairs which are mutual neighbors
        """
        print("Applying solution of Procrustes problem to get better mapping...")
        proc_dict = {self.generator.src_ph: final_pairs[:, 0],
                     self.generator.tgt_ph: final_pairs[:, 1]}
        A, B = sess.run([self.generator.src_emb,
                         self.generator.tgt_emb],
                        feed_dict=proc_dict)
        # pylint: disable=no-member
        R = scipy.linalg.orthogonal_procrustes(A, B)
        sess.run(tf.assign(self.generator.W, R[0]))

    def generate_xling_embed(self, sess, src_dict, tgt_dict, tgt_vec):
        """
        Generates cross lingual embeddings
        Arguments:
             sess(tf.session): Tensorflow session
        """
        print("Generating Cross-lingual embeddings...")
        src_emb_x = []
        batch_size = 512
        for i in range(0, self.vocab_size, batch_size):
            sids = [x for x in range(i, min(i + batch_size, self.vocab_size))]
            src_emb_x.append(sess.run(self.generator.mapWX,
                                      feed_dict={self.generator.src_ph: sids}))
        src_emb_x = np.concatenate(src_emb_x)
        print("Writing cross-lingual embeddings to file...")
        src_path = os.path.join(self.save_dir, "vectors-%s.txt" % self.slang)
        tgt_path = os.path.join(self.save_dir, "vectors-%s.txt" % self.tlang)
        with io.open(src_path, "w", encoding="utf-8") as f:
            f.write(u"%i %i\n" % src_emb_x.shape)
            for i in range(len(src_dict)):
                f.write(u"%s %s\n" % (src_dict[i], " ".join('%.5f' % x for x in src_emb_x[i])))

        with io.open(tgt_path, "w", encoding="utf-8") as f:
            f.write(u"%i %i\n" % tgt_vec.shape)
            for i in range(len(tgt_dict)):
                f.write(u"%s %s\n" % (tgt_dict[i], " ".join('%.5f' % x for x in tgt_vec[i])))
