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
Interactive mode for a trained End-to-End Memory Network on the Facebook bAbI
goal-oriented dialog dataset.

Reference:
    "Learning End-to-End Goal Oriented Dialog"
    https://arxiv.org/abs/1605.07683.

Usage:

    python interactive.py --task 5 --model_file memn2n_weights.npz

    Note: Ensure you specify the same parameters during inference as used to train.

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
from ngraph.frontends.neon import Layer
from ngraph.frontends.neon import GaussianInit, Adam
from ngraph.frontends.neon import make_bound_computation
from ngraph.frontends.neon import NgraphArgparser
from ngraph.frontends.neon import ArrayIterator
from ngraph.frontends.neon import Saver
import ngraph.transformers as ngt
from nlp_architect.data.babi_dialog import BABI_Dialog
from nlp_architect.models.memn2n_dialogue import MemN2N_Dialog
from utils import interactive_loop
from nlp_architect.utils.io import validate_existing_filepath, validate_parent_exists, validate

# parse the command line arguments
parser = NgraphArgparser(__doc__)
parser.add_argument(
    '--task',
    type=int,
    default='1',
    choices=range(1, 7),
    help='the task ID to train/test on from bAbI-dialog dataset (1-6)')
parser.add_argument(
    '--emb_size',
    type=int,
    default='32',
    help='Size of the word-embedding used in the model. (default 128)')
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
    help='load cached vectorized data')
parser.add_argument(
    '--use_oov',
    default=False,
    action='store_true',
    help='use OOV test set')
parser.add_argument(
    '--eps',
    type=float,
    default=1e-8,
    help='epsilon used to avoid divide by zero in softmax renormalization.',
    action=check_size(1e-100,1e-2))
parser.add_argument(
    '--model_file',
    default='memn2n_weights.npz',
    help='File to load model weights from.',
    type=str)

parser.set_defaults(batch_size=32, epochs=200)
args = parser.parse_args()

validate((args.emb_size, int, 1, 10000),
         (args.eps, float, 1e-15, 1e-2))

# Sanitize inputs
validate_existing_filepath(args.model_file)
model_file = args.model_file
assert model_file.endswith('.npz')
validate_parent_exists(args.data_dir)
data_dir = args.data_dir

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

a_pred, attention = memn2n(inputs)

# specify loss function, calculate loss and update weights
loss = ng.cross_entropy_multi(a_pred, inputs['answer'], usebits=True)

mean_cost = ng.sum(loss, out_axes=[])
optimizer = Adam(learning_rate=0.001)
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

with closing(ngt.make_transformer()) as transformer:
    interactive_computation = make_bound_computation(
        transformer, interactive_outputs, inputs)

    # Restore weights
    weight_saver.setup_restore(
        transformer=transformer,
        computation=train_outputs,
        filename=model_file)
    weight_saver.restore()

    # Add interactive mode
    print("Beginning interactive mode...")
    interactive_loop(interactive_computation, babi)
