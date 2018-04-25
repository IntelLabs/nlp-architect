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
from contextlib import closing
import ngraph as ng
from ngraph.frontends.neon import Layer
from ngraph.frontends.neon import Adam
from ngraph.frontends.neon import make_bound_computation
from ngraph.frontends.neon import NgraphArgparser
from ngraph.frontends.neon import ArrayIterator
from ngraph.frontends.neon import Saver
import ngraph.transformers as ngt

from models.kvmemn2n.model import KVMemN2N
from models.kvmemn2n.data import WIKIMOVIES
from models.kvmemn2n.interactive_util import interactive_loop

from tqdm import tqdm
import numpy as np
import os


# parse the command line arguments
parser = NgraphArgparser(__doc__)
parser.add_argument('--emb_size', type=int, default='50',
                    help='Size of the word-embedding used in the model. (default 50)')
parser.add_argument('--nhops', type=int, default='3',
                    help='Number of memory hops in the network')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--subset', type=str, default='wiki-entities',
                    choices=['full', 'wiki-entities'],
                    help='wikiMovies dataset to use for training examples.')
parser.add_argument('--reparse', action="store_true", default=False,
                    help='redo the data preprocessing')
parser.add_argument('--mem_mode', type=str, default='kb', choices=['kb', 'text'],
                    help='the memory source for the model, either knowledge base or text')
parser.add_argument('--use_v_luts', action="store_true",
                    help="Run the model using separate value lookup tables for each hop")
parser.add_argument('--model_file', type=str, default=None,
                    help="File to save or load weights from")
parser.add_argument('--inference', action="store_true", help="Run Inference with loaded weight")
parser.add_argument('--restore', action="store_true",
                    help="Run the model restoring weights from model_file")
parser.add_argument('--interactive', action="store_true",
                    help="Run Inference on User-supplied text either after training or \
                    with saved weights")


parser.set_defaults()
args = parser.parse_args()

if (args.inference is True) and (args.model_file is None):
    print("Need to set --model_file for Inference problem")
    quit()

if(args.model_file is not None):
    model_file = os.path.expanduser(args.model_file)
else:
    model_file = None

wikimovies = WIKIMOVIES(args.data_dir, subset=args.subset, reparse=args.reparse,
                        mem_source=args.mem_mode)

ndata = wikimovies.data_dict['train']['query']['data'].shape[0]
num_iterations = ndata // args.batch_size

train_set = ArrayIterator(wikimovies.data_dict['train'], batch_size=args.batch_size,
                          total_iterations=num_iterations)
test_set = ArrayIterator(wikimovies.data_dict['test'], batch_size=args.batch_size)
inputs = train_set.make_placeholders()
vocab_axis = ng.make_axis(length=wikimovies.vocab_size, name='vocab_axis')

memn2n = KVMemN2N(num_iterations, args.batch_size, args.emb_size, args.nhops,
                  wikimovies.story_length, wikimovies.memory_size, wikimovies.vocab_size,
                  vocab_axis, args.use_v_luts)
# Compute answer predictions
a_pred, _ = memn2n(inputs)

loss = ng.cross_entropy_multi(a_pred, ng.one_hot(inputs['answer'], axis=vocab_axis), usebits=True)


mean_cost = ng.sum(loss, out_axes=[])

optimizer = Adam(learning_rate=args.lr)

updates = optimizer(loss)

batch_cost = ng.sequential([updates, mean_cost])

# provide outputs for bound computation
train_outputs = dict(batch_cost=batch_cost, train_preds=a_pred)

with Layer.inference_mode_on():
    a_pred_inference, _ = memn2n(inputs)
    eval_loss = ng.cross_entropy_multi(a_pred_inference,
                                       ng.one_hot(inputs['answer'], axis=vocab_axis), usebits=True)

eval_outputs = dict(test_cross_ent_loss=eval_loss, test_preds=a_pred_inference)

if args.interactive:
    interactive_outputs = dict(test_preds=a_pred_inference)

if (model_file is not None):
    # Instantiate the Saver object to save weights
    weight_saver = Saver()

if (args.inference is False):
    # Train Loop
    with closing(ngt.make_transformer()) as transformer:
        # bind the computations
        train_computation = make_bound_computation(transformer, train_outputs, inputs)
        eval_computation = make_bound_computation(transformer, eval_outputs, inputs)

        if(model_file is not None and args.restore):
            weight_saver.setup_restore(transformer=transformer, computation=train_outputs,
                                       filename=model_file)
            # Restore weight
            weight_saver.restore()
        if(model_file is not None):
            weight_saver.setup_save(transformer=transformer, computation=train_outputs)

        for e in range(args.epochs+1):
            train_error = []
            train_loss = []
            for idx, data in tqdm(enumerate(train_set)):
                train_output = train_computation(data)
                niter = idx + 1
                if niter % args.iter_interval == 0:
                    preds = np.argmax(train_output['train_preds'], axis=0)
                    error = np.mean(data['answer'] != preds)
                    print("\nIteration {}, train_loss {}, train_batch_error {}".
                          format(niter, train_output['batch_cost'], error))

            test_error = []
            test_loss = []
            for idx, data in enumerate(test_set):
                test_output = eval_computation(data)
                test_loss.append(np.sum(test_output['test_cross_ent_loss']))
                preds = np.argmax(test_output['test_preds'], axis=0)
                error = np.mean(data['answer'] != preds)
                test_error.append(error)
            print("\ Epoch {}, Test_loss {}, test_batch_error {}".format(e, np.mean(test_loss),
                  np.mean(test_error)))
            # Shuffle training set and reset the others
            shuf_idx = np.random.permutation(range(train_set.data_arrays['query'].shape[0]))
            train_set.data_arrays = {k: v[shuf_idx] for k, v in train_set.data_arrays.items()}
            train_set.reset()
            test_set.reset()

            if(model_file is not None and e % 50 == 0):
                print('Saving model to: ', model_file)
                weight_saver.save(filename=model_file)
else:
    print('Loading saved model')
    with closing(ngt.make_transformer()) as transformer:
        eval_computation = make_bound_computation(transformer, eval_outputs, inputs)
        if (args.interactive):
            interactive_computation = make_bound_computation(transformer, interactive_outputs, inputs)
        weight_saver.setup_restore(transformer=transformer, computation=eval_outputs,
                                   filename=model_file)
        # Restore weight
        weight_saver.restore()

        test_error = []
        test_loss = []
        for idx, data in enumerate(test_set):
            test_output = eval_computation(data)
            test_loss.append(np.sum(test_output['test_cross_ent_loss']))
            preds = np.argmax(test_output['test_preds'], axis=0)
            error = np.mean(data['answer'] != preds)
            test_error.append(error)
        print("\Test_loss {}, test_batch_error {}".format(np.mean(test_loss), np.mean(test_error)))

if (args.interactive):
    with closing(ngt.make_transformer()) as transformer:
        print("Beginning interactive mode...")
        # Begin interactive loop
        interactive_loop(interactive_computation, wikimovies, args.batch_size)
