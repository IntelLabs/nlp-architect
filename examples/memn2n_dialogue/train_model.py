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
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from contextlib import closing
import ngraph as ng
from nlp_architect.data.babi_dialog import BABI_Dialog
from ngraph.frontends.neon import Layer
from ngraph.frontends.neon import GaussianInit, Adam
from ngraph.frontends.neon import make_bound_computation
from ngraph.frontends.neon import NgraphArgparser
from ngraph.frontends.neon import ArrayIterator
from ngraph.frontends.neon import Saver
import ngraph.transformers as ngt
from nlp_architect.models.memn2n_dialogue import MemN2N_Dialog
import numpy as np
import os
from tqdm import tqdm
from utils import interactive_loop
from nlp_architect.utils.io import validate_parent_exists, check_size, validate


# parse the command line arguments
parser = NgraphArgparser(__doc__)
parser.add_argument(
    '--task',
    type=int,
    default='1',
    choices=range(1,7),
    help='the task ID to train/test on from bAbI-dialog dataset (1-6)')
parser.add_argument(
    '--emb_size',
    type=int,
    default='32',
    help='Size of the word-embedding used in the model.')
parser.add_argument(
    '--nhops',
    type=int,
    default='3',
    help='Number of memory hops in the network',
    choices=range(1,10))
parser.add_argument(
    '--use_match_type',
    default=False,
    action='store_true',
    help='use match type features')
parser.add_argument(
    '--cache_match_type',
    default=False,
    action='store_true',
    help='cache match type answers')
parser.add_argument(
    '--cache_vectorized',
    default=False,
    action='store_true',
    help='cache vectorized data')
parser.add_argument(
    '--use_oov',
    default=False,
    action='store_true',
    help='use OOV test set')
parser.add_argument(
    '--lr',
    type=float,
    default=0.001,
    help='learning rate')
parser.add_argument(
    '--grad_clip_norm',
    type=float,
    default=40.0,
    help='Clip gradients such that norm is below this value.')
parser.add_argument(
    '--eps',
    type=float,
    default=1e-8,
    help='epsilon used to avoid divide by zero in softmax renormalization.')
parser.add_argument(
    '--save_log',
    action='store_true',
    default=False,
    help='Save evaluation results to log file.')
parser.add_argument(
    '--log_file',
    type=str,
    default='memn2n_dialgoue_results.txt',
    help='File to write evaluation set results to.')
parser.add_argument(
    '--weights_save_path',
    type=str,
    default='memn2n_weights.npz',
    help='File to save model weights to.')
parser.add_argument(
    '--save_epochs',
    type=int,
    default=1,
    help='Number of epochs between saving model weights.',
    action=check_size(1, 1000))
parser.add_argument(
    '--restore',
    default=False,
    action='store_true',
    help='Restore weights if found.')
parser.add_argument(
    '--interactive',
    default=False,
    action='store_true',
    help='enable interactive mode at the end of training.')
parser.add_argument(
    '--test',
    default=False,
    action='store_true',
    help='evaluate on the test set at the end of training.')

parser.set_defaults(batch_size=32, epochs=200)
args = parser.parse_args()

validate((args.emb_size, int, 1, 10000),
         (args.eps, float, 1e-15, 1e-2),
         (args.lr, float, 1e-8, 10),
         (args.grad_clip_norm, float, 1e-3, 1e5))

# Validate inputs
validate_parent_exists(args.log_file)
log_file = args.log_file
validate_parent_exists(args.weights_save_path)
weights_save_path = args.weights_save_path
validate_parent_exists(args.data_dir)
data_dir = args.data_dir
assert weights_save_path.endswith('.npz')
assert log_file.endswith('.txt')

gradient_clip_norm = args.grad_clip_norm

babi = BABI_Dialog(
    path=data_dir,
    task=args.task,
    oov=args.use_oov,
    use_match_type=args.use_match_type,
    cache_match_type=args.cache_match_type,
    cache_vectorized=args.cache_vectorized)

weight_saver = Saver()

# Set num iterations to 1 epoch since we loop over epochs & shuffle
ndata = babi.data_dict['train']['memory']['data'].shape[0]
num_iterations = ndata // args.batch_size

train_set = ArrayIterator(babi.data_dict['train'], batch_size=args.batch_size,
                          total_iterations=num_iterations)
dev_set = ArrayIterator(babi.data_dict['dev'], batch_size=args.batch_size)
test_set = ArrayIterator(babi.data_dict['test'], batch_size=args.batch_size)
inputs = train_set.make_placeholders()

memn2n = MemN2N_Dialog(
    babi.cands,
    babi.num_cands,
    babi.max_cand_len,
    babi.memory_size,
    babi.max_utt_len,
    babi.vocab_size,
    args.emb_size,
    args.batch_size,
    use_match_type=args.use_match_type,
    kb_ents_to_type=babi.kb_ents_to_type,
    kb_ents_to_cand_idxs=babi.kb_ents_to_cand_idxs,
    match_type_idxs=babi.match_type_idxs,
    nhops=args.nhops,
    eps=args.eps,
    init=GaussianInit(
        mean=0.0,
        std=0.1))

# Compute answer predictions
a_pred, attention = memn2n(inputs)

# specify loss function, calculate loss and update weights
loss = ng.cross_entropy_multi(a_pred, inputs['answer'], usebits=True)

mean_cost = ng.sum(loss, out_axes=[])
optimizer = Adam(learning_rate=args.lr)
updates = optimizer(loss)

batch_cost = ng.sequential([updates, mean_cost])

# provide outputs for bound computation
train_outputs = dict(batch_cost=batch_cost, train_preds=a_pred)

with Layer.inference_mode_on():
    a_pred_inference, attention_inference = memn2n(inputs)
    eval_loss = ng.cross_entropy_multi(
        a_pred_inference, inputs['answer'], usebits=True)

interactive_outputs = dict(
    test_preds=a_pred_inference,
    attention=attention_inference)
eval_outputs = dict(test_cross_ent_loss=eval_loss, test_preds=a_pred_inference)

# Train Loop
with closing(ngt.make_transformer()) as transformer:
    # bind the computations
    train_computation = make_bound_computation(
        transformer, train_outputs, inputs)
    loss_computation = make_bound_computation(
        transformer, eval_outputs, inputs)
    interactive_computation = make_bound_computation(
        transformer, interactive_outputs, inputs)

    weight_saver.setup_save(transformer=transformer, computation=train_outputs)

    if args.restore and os.path.exists(weights_save_path):
        print("Loading weights from {}".format(weights_save_path))
        weight_saver.setup_restore(
            transformer=transformer,
            computation=train_outputs,
            filename=weights_save_path)
        weight_saver.restore()
    elif args.restore and os.path.exists(weights_save_path) is False:
        print("Could not find weights at {}. ".format(weights_save_path)
              + "Running with random initialization.")

    for e in range(args.epochs):
        train_error = []
        train_cost = []
        for idx, data in enumerate(
            tqdm(train_set, total=train_set.nbatches,
                 unit='minibatches', desc="Epoch {}".format(e))):
            train_output = train_computation(data)
            train_cost.append(train_output['batch_cost'])
            preds = np.argmax(train_output['train_preds'], axis=1)
            error = np.mean(data['answer'].argmax(axis=1) != preds)
            train_error.append(error)

        train_cost_str = "Epoch {}: train_cost {}, train_error {}".format(
            e, np.mean(train_cost), np.mean(train_error))
        print(train_cost_str)
        if args.save_log:
            with open(log_file, 'a') as f:
                f.write(train_cost_str + '\n')

        if e % args.save_epochs == 0:
            print("Saving model to {}".format(weights_save_path))
            weight_saver.save(filename=weights_save_path)
            print("Saving complete")

        # Eval after each epoch
        test_loss = []
        test_error = []
        for idx, data in enumerate(dev_set):
            test_output = loss_computation(data)
            test_loss.append(np.sum(test_output['test_cross_ent_loss']))
            preds = np.argmax(test_output['test_preds'], axis=1)
            error = np.mean(data['answer'].argmax(axis=1) != preds)
            test_error.append(error)

        val_cost_str = "Epoch {}: validation_cost {}, validation_error {}".format(
            e, np.mean(test_loss), np.mean(test_error))
        print(val_cost_str)
        if args.save_log:
            with open(log_file, 'a') as f:
                f.write(val_cost_str + '\n')

        # Shuffle training set and reset the others
        shuf_idx = np.random.permutation(
            range(train_set.data_arrays['memory'].shape[0]))
        train_set.data_arrays = {k: v[shuf_idx]
                                 for k, v in train_set.data_arrays.items()}
        train_set.reset()
        dev_set.reset()

    print('Training Complete.')

    if args.interactive:
        interactive_loop(interactive_computation, babi)

    if args.test:
        # Final evaluation on test set
        test_loss = []
        test_error = []
        for idx, data in enumerate(test_set):
            test_output = loss_computation(data)
            test_loss.append(np.sum(test_output['test_cross_ent_loss']))
            preds = np.argmax(test_output['test_preds'], axis=1)
            error = np.mean(data['answer'].argmax(axis=1) != preds)
            test_error.append(error)

        test_cost_str = "test_cost {}, test_error {}".format(
             np.mean(test_loss), np.mean(test_error))
        print(test_cost_str)
        if args.save_log:
            with open(log_file, 'a') as f:
                f.write(test_cost_str + '\n')
