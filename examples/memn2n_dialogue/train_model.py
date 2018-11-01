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
Example that trains an End-to-End Memory Network on the Facebook bAbI
goal-oriented dialog dataset.

Reference:
    "Learning End-to-End Goal Oriented Dialog"
    https://arxiv.org/abs/1605.07683.

Usage:

    python train_memn2n.py --task 5

    use --task to specify which bAbI-dialog task to run on
        - Task 1: Issuing API Calls
        - Task 2: Updating API Calls
        - Task 3: Displaying Options
        - Task 4: Providing Extra Information
        - Task 5: Conducting Full Dialogs
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
import tensorflow as tf
from interactive_utils import interactive_loop
from tqdm import tqdm

from nlp_architect.data.babi_dialog import BABI_Dialog
from nlp_architect.models.memn2n_dialogue import MemN2N_Dialog
from nlp_architect.utils.io import validate_parent_exists, validate

# parse the command line arguments
tf.flags.DEFINE_integer(
    'task',
    1,
    'the task ID to train/test on from bAbI-dialog dataset (1-6)')
tf.flags.DEFINE_integer(
    'emb_size',
    20,
    'Size of the word-embedding used in the model.')
tf.flags.DEFINE_integer(
    'batch_size',
    32,
    'Size of the batch for optimization.')
tf.flags.DEFINE_integer(
    'nhops',
    3,
    'Number of memory hops in the network')
tf.flags.DEFINE_boolean(
    'use_match_type',
    False,
    'use match type features')
tf.flags.DEFINE_boolean(
    'cache_match_type',
    False,
    'cache match type answers')
tf.flags.DEFINE_boolean(
    'cache_vectorized',
    False,
    'cache vectorized data')
tf.flags.DEFINE_boolean(
    'use_oov',
    False,
    'use OOV test set')
tf.flags.DEFINE_float(
    'lr',
    0.001,
    'learning rate')
tf.flags.DEFINE_float(
    'grad_clip_norm',
    40.0,
    'Clip gradients such that norm is below this value.')
tf.flags.DEFINE_float(
    'eps',
    1e-8,
    'epsilon used to avoid divide by zero in softmax renormalization.')
tf.flags.DEFINE_boolean(
    'save_log',
    False,
    'Save evaluation results to log file.')
tf.flags.DEFINE_string(
    'data_dir',
    'data/',
    'File to save model weights to.')
tf.flags.DEFINE_string(
    'log_file',
    'memn2n_dialgoue_results.txt',
    'File to write evaluation set results to.')
tf.flags.DEFINE_string(
    'weights_save_path',
    'saved_tf/',
    'File to save model weights to.')
tf.flags.DEFINE_integer(
    'save_epochs',
    10,
    'Number of epochs between saving model weights.')
tf.flags.DEFINE_integer(
    'epochs',
    100,
    'Number of epochs between saving model weights.')
tf.flags.DEFINE_boolean(
    'restore',
    False,
    'Restore weights if found.')
tf.flags.DEFINE_boolean(
    'interactive',
    False,
    'enable interactive mode at the end of training.')
tf.flags.DEFINE_boolean(
    'test',
    False,
    'evaluate on the test set at the end of training.')
FLAGS = tf.flags.FLAGS

# Validate inputs
validate((FLAGS.task, int, 1, 7),
         (FLAGS.nhops, int, 1, 10),
         (FLAGS.batch_size, int, 1, 32000),
         (FLAGS.emb_size, int, 1, 10000),
         (FLAGS.eps, float, 1e-15, 1e-2),
         (FLAGS.lr, float, 1e-8, 10),
         (FLAGS.grad_clip_norm, float, 1e-3, 1e5),
         (FLAGS.epochs, int, 1, 1e10),
         (FLAGS.save_epochs, int, 1, 1e10))

current_dir = os.path.dirname(os.path.realpath(__file__))
log_file = os.path.join(current_dir, FLAGS.log_file)
validate_parent_exists(log_file)
weights_save_path = os.path.join(current_dir, FLAGS.weights_save_path)
validate_parent_exists(weights_save_path)
data_dir = os.path.join(current_dir, FLAGS.data_dir)
validate_parent_exists(data_dir)
assert log_file.endswith('.txt')

babi = BABI_Dialog(
    path=data_dir,
    task=FLAGS.task,
    oov=FLAGS.use_oov,
    use_match_type=FLAGS.use_match_type,
    cache_match_type=FLAGS.cache_match_type,
    cache_vectorized=FLAGS.cache_vectorized)

train_set = babi.data_dict['train']
dev_set = babi.data_dict['dev']
test_set = babi.data_dict['test']

n_train = train_set['memory']['data'].shape[0]
n_val = dev_set['memory']['data'].shape[0]
n_test = test_set['memory']['data'].shape[0]

train_batches = zip(range(0, n_train - FLAGS.batch_size, FLAGS.batch_size),
                    range(FLAGS.batch_size, n_train, FLAGS.batch_size))
train_batches = [(start, end) for start, end in train_batches]

val_batches = zip(range(0, n_val - FLAGS.batch_size, FLAGS.batch_size),
                  range(FLAGS.batch_size, n_val, FLAGS.batch_size))
val_batches = [(start, end) for start, end in val_batches]

test_batches = zip(range(0, n_test - FLAGS.batch_size, FLAGS.batch_size),
                   range(FLAGS.batch_size, n_test, FLAGS.batch_size))
test_batches = [(start, end) for start, end in test_batches]

with tf.Session() as sess:
    memn2n = MemN2N_Dialog(
        FLAGS.batch_size,
        babi.vocab_size,
        babi.max_utt_len,
        babi.memory_size,
        FLAGS.emb_size,
        babi.num_cands,
        babi.max_cand_len,
        hops=FLAGS.nhops,
        max_grad_norm=FLAGS.grad_clip_norm,
        optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.lr, epsilon=FLAGS.eps),
        session=sess)

    if FLAGS.restore and os.path.exists(weights_save_path):
        print("Loading weights from {}".format(weights_save_path))
        memn2n.saver.restore(sess, weights_save_path)
    elif FLAGS.restore and os.path.exists(weights_save_path) is False:
        print("Could not find weights at {}. ".format(weights_save_path)
              + "Running with random initialization.")

    for e in range(FLAGS.epochs):
        np.random.shuffle(train_batches)
        train_cost = []

        for start, end in tqdm(train_batches, total=len(train_batches),
                               unit='minibatches', desc="Epoch {}".format(e)):
            s = train_set['memory']['data'][start:end]
            q = train_set['user_utt']['data'][start:end]
            a = train_set['answer']['data'][start:end]

            if not FLAGS.use_match_type:
                c = np.tile(np.expand_dims(babi.cands, 0), [FLAGS.batch_size, 1, 1])
            else:
                c = train_set['cands_mat']['data'][start:end]

            cost = memn2n.batch_fit(s, q, a, c)
            train_cost.append(cost)

        train_cost_str = "Epoch {}: train_cost {}".format(e, np.mean(train_cost))
        print(train_cost_str)

        if FLAGS.save_log:
            with open(log_file, 'a') as f:
                f.write(train_cost_str + '\n')

        if e % FLAGS.save_epochs == 0:
            print("Saving model to {}".format(weights_save_path))
            memn2n.saver.save(sess, weights_save_path)
            print("Saving complete")

            val_error = []
            # Eval after each epoch
            for start, end in tqdm(val_batches, total=len(val_batches),
                                   unit='minibatches', desc="Epoch {}".format(e)):
                s = dev_set['memory']['data'][start:end]
                q = dev_set['user_utt']['data'][start:end]
                a = dev_set['answer']['data'][start:end]

                if not FLAGS.use_match_type:
                    c = np.tile(np.expand_dims(babi.cands, 0), [FLAGS.batch_size, 1, 1])
                else:
                    c = dev_set['cands_mat']['data'][start:end]

                a_pred = memn2n.predict(s, q, c)

                error = np.mean(a.argmax(axis=1) != a_pred)
                val_error.append(error)

            val_err_str = "Epoch {}: Validation Error: {}".format(e, np.mean(val_error))
            print(val_err_str)
            if FLAGS.save_log:
                with open(log_file, 'a') as f:
                    f.write(val_err_str + '\n')

    print('Training Complete.')
    print("Saving model to {}".format(weights_save_path))
    memn2n.saver.save(sess, weights_save_path)
    print("Saving complete")

    if FLAGS.interactive:
        interactive_loop(memn2n, babi)

    if FLAGS.test:
        # Final evaluation on test set
        test_error = []
        # Eval after each epoch
        for start, end in tqdm(test_batches, total=len(test_batches),
                               unit='minibatches'):
            s = test_set['memory']['data'][start:end]
            q = test_set['user_utt']['data'][start:end]
            a = test_set['answer']['data'][start:end]

            if not FLAGS.use_match_type:
                c = np.tile(np.expand_dims(babi.cands, 0), [FLAGS.batch_size, 1, 1])
            else:
                c = test_set['cands_mat']['data'][start:end]

            a_pred = memn2n.predict(s, q, c)

            error = np.mean(a.argmax(axis=1) != a_pred)
            test_error.append(error)

        test_err_str = "Test Error: {}".format(np.mean(test_error))
        print(test_err_str)
        if FLAGS.save_log:
            with open(log_file, 'a') as f:
                f.write(test_err_str + '\n')
