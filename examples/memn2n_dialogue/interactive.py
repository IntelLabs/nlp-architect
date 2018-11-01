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

import tensorflow as tf
from interactive_utils import interactive_loop

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
tf.flags.DEFINE_string(
    'data_dir',
    'data/',
    'File to save model weights to.')
tf.flags.DEFINE_string(
    'weights_save_path',
    'saved_tf/',
    'File to save model weights to.')
FLAGS = tf.flags.FLAGS


validate((FLAGS.task, int, 1, 7),
         (FLAGS.nhops, int, 1, 100),
         (FLAGS.emb_size, int, 1, 10000))

# Validate inputs
current_dir = os.path.dirname(os.path.realpath(__file__))
weights_save_path = os.path.join(current_dir, FLAGS.weights_save_path)
validate_parent_exists(weights_save_path)
data_dir = os.path.join(current_dir, FLAGS.data_dir)
validate_parent_exists(data_dir)

babi = BABI_Dialog(
    path=data_dir,
    task=FLAGS.task,
    oov=FLAGS.use_oov,
    use_match_type=FLAGS.use_match_type,
    cache_match_type=FLAGS.cache_match_type,
    cache_vectorized=FLAGS.cache_vectorized)

with tf.Session() as sess:
    memn2n = MemN2N_Dialog(
        32,
        babi.vocab_size,
        babi.max_utt_len,
        babi.memory_size,
        FLAGS.emb_size,
        babi.num_cands,
        babi.max_cand_len,
        hops=FLAGS.nhops,
        max_grad_norm=40.0,
        session=sess)

    if os.path.exists(weights_save_path):
        print("Loading weights from {}".format(weights_save_path))
        memn2n.saver.restore(sess, weights_save_path)
        print("Beginning interactive mode...")
        interactive_loop(memn2n, babi)
    else:
        print("Could not find weights at {}. Exiting.".format(weights_save_path))
