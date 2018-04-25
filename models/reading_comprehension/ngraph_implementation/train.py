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
import ngraph as ng
from ngraph.frontends.neon import (Layer, Tanh,
                                   LSTM, Logistic)

from models.reading_comprehension.ngraph_implementation.layers import (
    MatchLSTMCell_withAttention,
    unroll_with_attention,
    AnswerPointer_withAttention,
    Dropout_Modified,
    LookupTable)

from ngraph.frontends.neon import (
    Adam,
    GlorotInit)
from ngraph.frontends.neon import (ax, make_bound_computation)
from ngraph.frontends.neon import NgraphArgparser
from ngraph.frontends.neon import ArrayIterator
import ngraph.transformers as ngt
import os
import numpy as np
from utils import (
    create_squad_training,
    max_values_squad,
    get_data_array_squad_ngraph,
    cal_f1_score)
import math
from weight_initilizers import (make_placeholder, make_weights)


"""
Model converges on gpu backened.
"""
# parse the command line arguments
parser = NgraphArgparser(__doc__)


parser.add_argument(
    '--data_path',
    help='enter path for training data')

parser.add_argument('--gpu_id', default="0",
                    help='enter gpu id')

parser.add_argument('--max_para_req', default=100,
                    help='enter the max length of paragraph')

parser.add_argument(
    '--batch_size_squad',
    default=16,
    help='enter the batch size')  # 16 is the max batch size o be fit on gpu

parser.set_defaults()

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

hidden_size = 150
gradient_clip_value = 15
embed_size = 300

params_dict = {}
params_dict['batch_size'] = args.batch_size_squad
params_dict['embed_size'] = 300
params_dict['pad_idx'] = 0
params_dict['hs'] = hidden_size
params_dict['glove_dim'] = 300
params_dict['iter_interval'] = 8000
params_dict['num_iterations'] = 500000
params_dict['ax'] = ax

# Initialzer
init = GlorotInit()
params_dict['init'] = init


path_gen = args.data_path

file_name_dict={}
file_name_dict['train_para_ids']='squad/train.ids.context'
file_name_dict['train_ques_ids']='squad/train.ids.question'
file_name_dict['train_answer']='squad/train.span'
file_name_dict['val_para_ids']='squad/dev.ids.context'
file_name_dict['val_ques_ids']='squad/dev.ids.question'
file_name_dict['val_ans']='squad/dev.span'
file_name_dict['vocab_file']='squad/vocab.dat'


train_para_ids = os.path.join(path_gen + file_name_dict['train_para_ids'])
train_ques_ids = os.path.join(path_gen + file_name_dict['train_ques_ids'])
answer_file = os.path.join(path_gen + file_name_dict['train_answer'])
val_paras_ids = os.path.join(path_gen + file_name_dict['val_para_ids'])
val_ques_ids = os.path.join(path_gen + file_name_dict['val_ques_ids'])
val_ans_file = os.path.join(path_gen + file_name_dict['val_ans'])
vocab_file = os.path.join(path_gen + file_name_dict['vocab_file'])


data_train, vocab_list = create_squad_training(train_para_ids, train_ques_ids,
                                               answer_file, vocab_file)

data_dev, _ = create_squad_training(val_paras_ids, val_ques_ids,
                                    val_ans_file, vocab_file)


data_total = data_train + data_dev
max_para, max_question = max_values_squad(data_total)
max_para = int(args.max_para_req)
print(max_question)
params_dict['max_question'] = max_question
params_dict['max_para'] = max_para

# chane this to experiment with smaller dataset sizes(eg 8000)
params_dict['train_set_size'] = len(data_train)
params_dict['vocab_size'] = len(vocab_list)

print('Loading Embeddings')
embeddingz = np.load(
    os.path.join(
        path_gen +
        "squad/glove.trimmed_zeroinit.300.npz"))
embeddings = embeddingz['glove']
vocab_file = os.path.join(path_gen + 'squad/vocab.dat')


print("creating training Set ")
train = get_data_array_squad_ngraph(params_dict, data_train, set_val='train')
dev = get_data_array_squad_ngraph(params_dict, data_dev, set_val='dev')
print('Train Set Size is', len(train['para']['data']))
print('Dev set size is', len(dev['para']['data']))


# Use Array Iterator for training set
train_set = ArrayIterator(train, batch_size=params_dict['batch_size'],
                          total_iterations=params_dict['num_iterations'])
# Use Array Iterator for validation set
valid_set = ArrayIterator(dev, batch_size=params_dict['batch_size'],
                          total_iterations=params_dict['num_iterations'])

# Make placeholderds for training
inputs = train_set.make_placeholders(include_iteration=True)


# Encoding Layer
rlayer_1 = LSTM(hidden_size, init, activation=Tanh(), reset_cells=True,
                gate_activation=Logistic(), return_sequence=True)

# Embedding Layer
embed_layer = LookupTable(
    params_dict['vocab_size'],
    params_dict['embed_size'],
    embeddings,
    update=False,
    pad_idx=params_dict['pad_idx'])


# Initialzers for LSTM Cells
input_placeholder, input_value = make_placeholder(
    2 * hidden_size, 1, params_dict['batch_size'])
input_placeholder_a, input_value = make_placeholder(
    2 * hidden_size, 1, params_dict['batch_size'])

W_in, W_rec, b, init_state, init_state_value = make_weights(
    input_placeholder, hidden_size, lambda w_axes: np.zeros(
        w_axes.lengths), lambda hidden_axis: np.ones(
            hidden_axis.length), None)

W_in_a, W_rec_a, b_a, init_state_a, init_state_value_a = make_weights(
    input_placeholder_a, hidden_size, lambda w_axes: np.zeros(
        w_axes.lengths), lambda hidden_axis: np.ones(
            hidden_axis.length), None)

# Initialize Match LSTM Cell
cell_init = MatchLSTMCell_withAttention(
    params_dict,
    hidden_size,
    init=W_in,
    init_h2h=W_rec,
    bias_init=b,
    activation=Tanh(),
    gate_activation=Logistic(),
    reset_cells=True)

# Initialize Answer Pointer Cell
answer_init = AnswerPointer_withAttention(
    params_dict,
    hidden_size,
    init=W_in_a,
    init_h2h=W_rec_a,
    bias_init=b_a,
    activation=Tanh(),
    gate_activation=Logistic(),
    reset_cells=True)


# Make  Required Axes
# Axis with length of batch size
N = ng.make_axis(length=params_dict['batch_size'], name='N')
# Axis with length of max question
REC = ng.make_axis(length=max_question, name='REC')
# Axis with length of hidden unit size
F = ng.make_axis(length=hidden_size, name='F')
# Axis with length of embedding size
F_embed = ng.make_axis(length=300, name='F_embed')
# Axis with length 1
dummy_axis = ng.make_axis(length=1, name='dummy_axis')
# Axis with length of answer span
span = ng.make_axis(length=2, name='span')


# Set up drop out layer
dropout_val = ng.slice_along_axis(inputs['dropout_val'], N, 0)
dropout_1 = Dropout_Modified(keep=dropout_val)
dropout_2 = Dropout_Modified(keep=dropout_val)
drop_pointer = ng.maximum(dropout_val, ng.constant(const=0.8, axes=[]))
dropout_3 = Dropout_Modified(keep=drop_pointer)
dropout_4 = Dropout_Modified(keep=drop_pointer)

# Constants required for masking
const_LSTM = ng.constant(axes=[F, dummy_axis], const=1)
const_loss = ng.constant(axes=[ax.Y, dummy_axis], const=1)
const_LSTM_embed = ng.constant(axes=[F_embed, dummy_axis], const=1)

# Create masks
reorder_para_mask = ng.axes_with_order(
    inputs['para_len'], axes=[
        dummy_axis, inputs['para_len'].axes[2], N])

reorder_ques_mask = ng.axes_with_order(
    inputs['question_len'], axes=[
        dummy_axis, inputs['question_len'].axes[2], N])

# Masks for question and para after encoding layer
mask_para = ng.dot(const_LSTM, reorder_para_mask)
mask_question = ng.dot(const_LSTM,
                       ng.cast_axes(reorder_ques_mask, [dummy_axis, REC, N]))

# Masks for question and para after embedding/LookupTable layer
mask_para_embed = ng.dot(const_LSTM_embed, reorder_para_mask)
mask_question_embed = ng.dot(
    const_LSTM_embed, ng.cast_axes(
        reorder_ques_mask, [
            dummy_axis, REC, N]))

# Pass question and para through embedding layer and dropout layers
embed_output_para_1 = embed_layer(inputs['para'])
embed_output_para = dropout_1(embed_output_para_1, keep=dropout_val)
question_inps = ng.cast_axes(inputs['question'], [N, REC])
embed_output_ques_1 = embed_layer(question_inps)
embed_output_ques = dropout_2(embed_output_ques_1, keep=dropout_val)


# Mask output of embedding layer to the length of each sentence
embed_output_para = ng.multiply(embed_output_para, ng.cast_axes(
    mask_para_embed, [embed_output_para.axes[0], embed_output_para.axes[1], N]))

embed_output_ques = ng.multiply(
    embed_output_ques, ng.cast_axes(
        mask_question_embed, [
            embed_output_ques.axes[0], embed_output_ques.axes[1], N]))


# Encoding Lyer
H_pr_be = rlayer_1(embed_output_ques)
H_hy_be = rlayer_1(embed_output_para)


# Mask the output of eencoding layers to the length of each sentence
H_hy = ng.multiply(H_hy_be, mask_para)
H_pr = ng.multiply(H_pr_be, mask_question)

# Unroll with attention in the forward direction
outputs_forward = unroll_with_attention(
    cell_init,
    max_para,
    H_pr,
    H_hy,
    init_states=None,
    reset_cells=True,
    return_sequence=True,
    reverse_mode=False,
    input_data=inputs)

# Unroll with attention in the reverse direction
outputs_reverse = unroll_with_attention(
    cell_init,
    max_para,
    H_pr,
    H_hy,
    init_states=None,
    reset_cells=True,
    return_sequence=True,
    reverse_mode=True,
    input_data=inputs)

# Mask unrolled outputs to the length of each sentence
outputs_forward_1 = ng.multiply(outputs_forward, mask_para)
outputs_reverse_1 = ng.multiply(outputs_reverse, mask_para)

# Dropout layer for each of the unrolled outputs
outputs_forward = dropout_3(outputs_forward_1, keep=drop_pointer)
outputs_reverse = dropout_4(outputs_reverse_1, keep=drop_pointer)
outputs_final = ng.concat_along_axis([outputs_forward, outputs_reverse],
                                     axis=outputs_reverse.axes.feature_axes()[0])

# Answer pointer pass
logits_concat = answer_init(outputs_final, states=None, output=None,
                            reset_cells=True, input_data=inputs)

# Logits
logits1 = ng.cast_axes(logits_concat[0], [ax.Y, N])
logits2 = ng.cast_axes(logits_concat[1], [ax.Y, N])

# Compute loss function
label1 = ng.slice_along_axis(
    inputs['answer'],
    axis=inputs['answer'].axes.feature_axes()[0],
    idx=0)

label2 = ng.slice_along_axis(
    inputs['answer'],
    axis=inputs['answer'].axes.feature_axes()[0],
    idx=1)
labels_concat = [label1, label2]
loss1 = ng.cross_entropy_multi(logits1,
                               ng.one_hot(label1, axis=ax.Y), usebits=False)

loss2 = ng.cross_entropy_multi(logits2,
                               ng.one_hot(label2, axis=ax.Y), usebits=False)

# Total Loss
train_loss = loss1 + loss2

# Set optimizer (no learning rate scheduler used)
optimizer = Adam(learning_rate=2e-3)


print('compiling the graph')
# Cost set up
batch_cost = ng.sequential(
    [optimizer(train_loss), ng.mean(train_loss, out_axes=())])

# Predicted class is the max probability out of the 2=3
# Required Outputs- Batch Cost, Train Probability,misclass train
train_outputs = dict(batch_cost=batch_cost, inps=inputs['answer'],
                     logits=ng.stack(logits_concat, span, 1),
                     labels=inputs['answer'], drop=dropout_val)

# Inference Mode for validation dataset:
with Layer.inference_mode_on():
    eval_outputs = dict(logits=ng.stack(logits_concat, span, 1),
                        labels=inputs['answer'], drop=drop_pointer)


# Now bind the computations we are interested in
print('generating transformer')
eval_frequency = 20
val_frequency = np.ceil(len(train['para']['data']) / params_dict['batch_size'])
train_error_frequency = 1000

# Create Transformer
transformer = ngt.make_transformer()
train_computation = make_bound_computation(transformer, train_outputs, inputs)
valid_computation = make_bound_computation(transformer, eval_outputs, inputs)


'''
TODO: Include feature to Save and load weights
'''

# Start Itearting through
epoch_no = 0
for idx, data in enumerate(train_set):
    train_output = train_computation(data)
    predictions = train_output['logits']
    label_batch = train_output['labels']
    niter = idx + 1
    # Print training loss and F-1 and EM scores after every 20 iterations
    if (idx % 20 == 0):

        print('iteration = {}, train loss = {}'.format(
            niter, train_output['batch_cost']))
        f1_score_int, em_score_int = cal_f1_score(
            params_dict, label_batch, predictions)
        print("F1_Score and EM_score are", f1_score_int, em_score_int)

    divide_val = math.ceil(
        len(dev['para']['data']) / params_dict['batch_size'])

    if niter % val_frequency == 0:
        print('Epoch done:', epoch_no)
        epoch_no += 1
        f1_score_req = 0
        em_score_req = 0

        # Compute validation scores
        for idx_val, data_val in enumerate(valid_set):
            eval_output = valid_computation(data_val)
            predictions_val = eval_output['logits']
            label_val = eval_output['labels']
            # Compute F1 and EM scores
            f1_score_val, em_score_val = cal_f1_score(
                params_dict, label_val, predictions_val)
            f1_score_req += f1_score_val
            em_score_req += em_score_val

            if (idx_val + 1) % divide_val == 0:
                break

        f1_val = f1_score_req / divide_val
        em_val = em_score_req / divide_val

        print('Computing validation Scores')
        print('Validation F1 and EM', f1_val, em_val)
