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
from ngraph.frontends.neon import (
    Layer,
    Affine,
    BaseRNNCell,
    Linear,
    get_steps,
    ax)
from ngraph.frontends.neon.graph import SubGraph
from ngraph.frontends.neon.axis import shadow_axes_map
import numpy as np


class MatchLSTMCell_withAttention(BaseRNNCell):
    """
    MatchLSTM cell fused with attention.

    Arguments:
    ----------
    params_dict : Dictionary containing information and hyperparameters
                      having the following fields:
                      (initialzer function,max length of para and max length of question)
    nout (int): Number of hidden/output units
    init (Initializer): Function to initialize the input-to-hidden weights.
        By default, this initializer will also be used to initialize recurrent
        weights unless init_inner is also specified. Biases are always
        initialized to zero.
    init_h2h (Initializer, optional): Function to initialize recurrent weights.
        If absent, will default to using the initializer passed as the init
        argument.
    activation (Transform): Activation function used to produce outputs.
    gate_activation (Transform): Activation function for gate inputs.
    batch_norm (bool, optional): Defaults to False. If True, batch normalization
        is applied to the weighted inputs.

    reset_cells: Reset cells if assigned to True

    Return Vale:
    -----------
    Returns the next state for the unrolled time step and  the previous states
    """

    def __init__(
            self,
            params_dict,
            nout,
            init,
            init_h2h=None,
            bias_init=None,
            activation=None,
            gate_activation=None,
            batch_norm=False,
            reset_cells=True,
            **kwargs):
        super(MatchLSTMCell_withAttention, self).__init__(**kwargs)

        self.init = params_dict['init']
        max_question = params_dict['max_question']
        max_para = params_dict['max_para']
        hidden_size = nout

        # Axes
        # Axis for length of the hidden units
        self.hidden_rows = ng.make_axis(length=hidden_size, name='hidden_rows')
        # Axis for length of the hidden units
        self.F = ng.make_axis(length=hidden_size, name='F')
        # Axis for length of max question length
        self.hidden_cols_ques = ng.make_axis(
            length=max_question, name='hidden_cols_ques')
        # Axis with length of embedding sizes
        self.embed_axis = ng.make_axis(
            length=params_dict['embed_size'],
            name='embed_axis')
        # Recurrent axis for max question length
        self.REC = ng.make_axis(length=max_question, name='REC')
        # axis with size 1
        self.dummy_axis = ng.make_axis(length=1, name='dummy_axis')
        # Axis for batch size
        self.N = ng.make_axis(length=params_dict['batch_size'], name='N')
        # Axis for the output of match lstm cell
        self.lstm_feature = ng.make_axis(
            length=2 * hidden_size, name='lstm_feature')
        # Length of final classification layer (maximum length of the
        # paragraph)
        self.ax = params_dict['ax']
        self.ax.Y.length = max_para

        # Variables to be learnt during training (part of the attention network)
        # naming convention taken from teh paper
        self.W_p = ng.variable(
            axes=[
                self.hidden_rows,
                self.F],
            initial_value=self.init)
        self.W_q = ng.variable(
            axes=[
                self.hidden_rows,
                self.F],
            initial_value=self.init)
        self.W_r = ng.variable(
            axes=[
                self.hidden_rows,
                self.F],
            initial_value=self.init)
        self.b_p = ng.variable(axes=self.hidden_rows, initial_value=self.init)
        self.w_lr = ng.variable(
            axes=[
                self.hidden_rows],
            initial_value=self.init)

        # Constants for creating masks and initial hidden states
        self.e_q = ng.constant(
            axes=[self.dummy_axis, self.hidden_cols_ques], const=np.ones([1, max_question]))
        self.e_q2 = ng.constant(axes=[self.F, self.dummy_axis], const=1)
        self.h_r_old = ng.constant(axes=[self.F, self.N], const=0)

        # Define variables for implementing the stacking operation. the default
        # stack op seems to be slow
        L1 = np.vstack(
            (np.eye(hidden_size), np.zeros([hidden_size, hidden_size])))
        L2 = np.vstack(
            (np.zeros([hidden_size, hidden_size]), np.eye(hidden_size)))
        self.ZX = ng.constant(const=L1, axes=[self.lstm_feature, self.F])
        self.ZY = ng.constant(const=L2, axes=[self.lstm_feature, self.F])

        # LSTM Cell Initialization (Code from the standard LSTM Cell in ngraph)
        self.nout = nout
        self.init = init
        self.init_h2h = init_h2h if init_h2h is not None else init
        self.bias_init = bias_init
        self.activation = activation
        if gate_activation is not None:
            self.gate_activation = gate_activation
        else:
            self.gate_activation = self.activation
        self.batch_norm = batch_norm
        self.reset_cells = reset_cells
        self.i2h = {}
        self.h2h = {}
        self.gate_transform = {}
        self.gate_output = {}
        for gate in self._gate_names:
            self.h2h[gate] = Linear(nout=self.nout,
                                    init=self.init_h2h[gate])
            self.i2h[gate] = Affine(axes=self.h2h[gate].axes,
                                    weight_init=self.init[gate],
                                    bias_init=self.bias_init[gate],
                                    batch_norm=self.batch_norm)
            if gate is 'g':
                self.gate_transform[gate] = self.activation
            else:
                self.gate_transform[gate] = self.gate_activation
        self.out_axes = None

    @property
    def state_info(self):
        return [{'state_name': 'h'},
                {'state_name': 'c'}]

    @property
    def feature_axes(self):
        return self.h2h['i'].axes.feature_axes()

    @property
    def _gate_names(self):
        return ['i', 'f', 'o', 'g']

    def __call__(
            self,
            H_pr,
            h_ip,
            states,
            output=None,
            reset_cells=True,
            input_data=None):
        """
        Arguments:
        ----------
        H_pr : Encoding for question
        h_ip: Sliced input of paragraph encoding for a particular time step
        states: State of the LSTM cell
        output: previous hidden state
        input_data: the ArrayIterator object for training data (contains information of
                                                        length of each sentence)
        """
        # get recurrent axis for question
        rec_axis_pr = H_pr.axes.recurrent_axis()
        const_one = ng.constant(const=1, axes=[self.dummy_axis])
        # if first word in a paragraph is encountered, assign the previous LSTM
        # hidden state as zeros
        if output is None:
            h_r_old = ng.constant(axes=[self.F, self.N], const=0)
        else:
            h_r_old = ng.cast_axes(output, [self.F, self.N])

        # Compute attention vector
        sum_1 = ng.dot(self.W_q, H_pr)
        sum_1 = ng.cast_axes(
            sum_1, [
                self.hidden_rows, self.hidden_cols_ques, self.N])
        int_sum1 = ng.dot(self.W_p, h_ip)
        int_sum2 = ng.dot(self.W_r, h_r_old)
        int_sum = int_sum1 + int_sum2 + self.b_p
        int_sum = ng.ExpandDims(int_sum, self.dummy_axis, 1)

        # making for the attention vector
        req_mask = ng.axes_with_order(
            ng.cast_axes(
                ng.dot(
                    self.e_q2, input_data['question_len']), [
                    self.hidden_rows, self.N, self.hidden_cols_ques]), [
                self.hidden_rows, self.hidden_cols_ques, self.N])

        req_mask_2 = ng.axes_with_order(
            ng.cast_axes(
                ng.dot(
                    const_one, input_data['question_len']), [
                    self.N, self.hidden_cols_ques]), [
                self.hidden_cols_ques, self.N])

        G_i_int = sum_1 + ng.multiply(req_mask,
                                      ng.axes_with_order(ng.dot(int_sum,
                                                                self.e_q),
                                                         [self.hidden_rows,
                                                          self.hidden_cols_ques,
                                                          self.N]))

        G_i = ng.tanh(G_i_int)
        # Attention Vector
        at_sum1 = ng.dot(self.w_lr, G_i)
        at = ng.softmax(at_sum1 + ng.log(req_mask_2))
        at_repeated = ng.cast_axes(
            ng.dot(
                self.e_q2, ng.ExpandDims(
                    at, self.dummy_axis, 0)), [
                self.F, rec_axis_pr, self.N])

        # Stack the 2 vectors as per the equation in the paper
        z1 = h_ip
        z2 = ng.sum(ng.multiply(H_pr, at_repeated), rec_axis_pr)
        # represents the inp to lstm_cell
        # ng.concat_along_axis([z1,z2],self.F)
        inputs_lstm = ng.dot(self.ZX, z1) + ng.dot(self.ZY, z2)

        # LSTM cell computations (from LSTM brach in ngraph)
        if self.out_axes is None:
            self.out_axes = self.feature_axes + inputs_lstm.axes.batch_axis()
        if states is None:
            states = self.initialize_states(inputs_lstm.axes.batch_axis(),
                                            reset_cells=reset_cells)
        assert self.out_axes == states['h'].axes

        for gate in self._gate_names:
            transform = self.gate_transform[gate]
            gate_input = self.i2h[gate](
                inputs_lstm) + self.h2h[gate](states['h'])
            self.gate_output[gate] = ng.cast_role(transform(gate_input),
                                                  self.out_axes)

        states['c'] = (states['c'] * self.gate_output['f']
                       + self.gate_output['i'] * self.gate_output['g'])
        states['h'] = self.gate_output['o'] * self.activation(states['c'])
        states['h'] = ng.cast_role(states['h'], self.out_axes)
        # return unrolled output and state of LSTM cell
        return ng.cast_axes(states['h'], axes=[self.F, self.N]), states


class AnswerPointer_withAttention(BaseRNNCell):
    """
    Answer pointer cell.

    Arguments:
    ----------
    params_dict : Dictionary containing information and hyperparameters
                  having the following fields:
                  (initialzer function,max length of para and max length of question)
    nout (int): Number of hidden/output units
    init (Initializer): Function to initialize the input-to-hidden weights.
        By default, this initializer will also be used to initialize recurrent
        weights unless init_inner is also specified. Biases are always
        initialized to zero.
    init_h2h (Initializer, optional): Function to initialize recurrent weights.
        If absent, will default to using the initializer passed as the init
        argument.
    activation (Transform): Activation function used to produce outputs.
    gate_activation (Transform): Activation function for gate inputs.
    batch_norm (bool, optional): Defaults to False. If True, batch normalization
        is applied to the weighted inputs.
    reset_cells: Reset cells if assigned to True

    Return Value:
    ----------------
    Returns a list of masked logits for the start and end indices
    """

    def __init__(
            self,
            params_dict,
            nout,
            init,
            init_h2h=None,
            bias_init=None,
            activation=None,
            gate_activation=None,
            batch_norm=False,
            reset_cells=True,
            **kwargs):
        super(AnswerPointer_withAttention, self).__init__(**kwargs)

        self.init_axes = params_dict['init']
        max_question = params_dict['max_question']
        max_para = params_dict['max_para']
        hidden_size = nout

        # Axes
        # Axis for length of the hidden units
        self.hidden_rows = ng.make_axis(length=hidden_size, name='hidden_rows')
        # Axis for length of max_para
        self.hidden_cols_para = ng.make_axis(
            length=max_para, name='hidden_cols_para')
        # Axis for length of hidden unit size
        self.F = ng.make_axis(length=hidden_size, name='F')
        # Axis for length of max_question
        self.REC = ng.make_axis(length=max_question, name='REC')
        # Axis with length 1
        self.dummy_axis = ng.make_axis(length=1, name='dummy_axis')
        # Axis with length of batch_size
        self.N = ng.make_axis(length=params_dict['batch_size'], name='N')
        # Axis with twice the length of hidden sizes
        self.lstm_feature_new = ng.make_axis(
            length=2 * hidden_size, name='lstm_feature')
        self.ax = params_dict['ax']
        # Length of final classification layer (maximum length of the
        # paragraph)
        self.ax.Y.length = max_para

        # Variables
        self.V_answer = ng.variable(
            axes=[
                self.hidden_rows,
                self.lstm_feature_new],
            initial_value=self.init_axes)
        self.W_a = ng.variable(
            axes=[
                self.hidden_rows,
                self.F],
            initial_value=self.init_axes)
        self.b_a = ng.variable(
            axes=self.hidden_rows,
            initial_value=self.init_axes)
        self.e_q = ng.constant(
            axes=[self.dummy_axis, self.hidden_cols_para], const=np.ones([1, max_para]))
        self.e_q2 = ng.constant(
            axes=[
                self.lstm_feature_new,
                self.dummy_axis],
            const=1)
        self.v_lr = ng.variable(
            axes=[
                self.hidden_rows],
            initial_value=self.init_axes)
        self.W_RNNx = ng.variable(
            axes=[
                self.hidden_rows,
                self.F],
            initial_value=self.init_axes)
        self.W_RNNh = ng.variable(
            axes=[
                self.hidden_rows,
                self.F],
            initial_value=self.init_axes)

        # LSTM Cell Initialization
        self.nout = nout
        self.init = init
        self.init_h2h = init_h2h if init_h2h is not None else init
        self.bias_init = bias_init
        self.activation = activation
        if gate_activation is not None:
            self.gate_activation = gate_activation
        else:
            self.gate_activation = self.activation
        self.batch_norm = batch_norm
        self.reset_cells = reset_cells
        self.i2h = {}
        self.h2h = {}
        self.gate_transform = {}
        self.gate_output = {}
        for gate in self._gate_names:
            self.h2h[gate] = Linear(nout=self.nout,
                                    init=self.init_h2h[gate])
            self.i2h[gate] = Affine(axes=self.h2h[gate].axes,
                                    weight_init=self.init[gate],
                                    bias_init=self.bias_init[gate],
                                    batch_norm=self.batch_norm)
            if gate is 'g':
                self.gate_transform[gate] = self.activation
            else:
                self.gate_transform[gate] = self.gate_activation
        self.out_axes = None

    @property
    def state_info(self):
        return [{'state_name': 'h'},
                {'state_name': 'c'}]

    @property
    def feature_axes(self):
        return self.h2h['i'].axes.feature_axes()

    @property
    def _gate_names(self):
        return ['i', 'f', 'o', 'g']

    def __call__(
            self,
            H_concat,
            states=None,
            output=None,
            reset_cells=True,
            input_data=None):
        """
        Arguments:
        ----------
        H_concat: Concatenated forward and reverse unrolled outputs of the
                 `MatchLSTMCell_withAttention` cell
        states: previous LSTM state
        output: hidden state from previous timestep
        reset_cells: argument to reset a cell
        input_data: the ArrayIterator object for training data
                    (contains information of length of each sentence)

        """

        rec_axis_pr = H_concat.axes.recurrent_axis()
        const_one = ng.constant(const=1, axes=[self.dummy_axis])

        b_k_lists = []
        # rec_axis_hy=H_hy.axes.recurrent_axis()
        for i in range(0, 2):
            if output is None:
                h_k_old = ng.constant(axes=[self.F, self.N], const=0)
            else:
                h_k_old = ng.cast_axes(output, [self.F, self.N])

            sum_1 = ng.dot(
                self.V_answer, ng.cast_axes(
                    H_concat, [
                        self.lstm_feature_new, rec_axis_pr, self.N]))
            sum_1 = ng.cast_axes(
                sum_1, [
                    self.hidden_rows, self.hidden_cols_para, self.N])

            int_sum2 = ng.dot(self.W_a, h_k_old)
            int_sum = int_sum2  # +self.b_a
            int_sum = ng.ExpandDims(int_sum, self.dummy_axis, 1)

            # Following notations from the paper
            # Compute Attention Vector
            F_i_int = sum_1 + ng.axes_with_order(
                ng.dot(
                    int_sum, self.e_q), [
                    self.hidden_rows, self.hidden_cols_para, self.N])

            F_i = ng.tanh(F_i_int)  # Attention Vector

            b_k_sum1 = ng.dot(self.v_lr, F_i)
            # This masking with -inf for length of para>max_para ensures that
            # when we do softmax over these values we get a 0
            mask_loss_new = ng.log(
                ng.dot(const_one,
                       input_data['para_len'])
            )
            mask_loss_new = ng.axes_with_order(
                ng.cast_axes(
                    mask_loss_new, [
                        self.N, self.hidden_cols_para]), [
                    self.hidden_cols_para, self.N])

            # Add mask to the required logits
            b_k = ng.softmax(b_k_sum1 + mask_loss_new)
            b_k_req = ng.softmax(b_k_sum1 + mask_loss_new)
            b_k_repeated = ng.cast_axes(
                ng.dot(
                    self.e_q2, ng.ExpandDims(
                        b_k, self.dummy_axis, 0)), [
                    H_concat.axes[0], rec_axis_pr, self.N])

            inputs_lstm = ng.sum(
                ng.multiply(
                    H_concat,
                    b_k_repeated),
                rec_axis_pr)

            # LSTM Cell calculations
            if self.out_axes is None:
                self.out_axes = self.feature_axes + inputs_lstm.axes.batch_axis()
            if states is None:
                states = self.initialize_states(inputs_lstm.axes.batch_axis(),
                                                reset_cells=reset_cells)
            assert self.out_axes == states['h'].axes

            for gate in self._gate_names:
                transform = self.gate_transform[gate]
                gate_input = self.i2h[gate](
                    inputs_lstm) + self.h2h[gate](states['h'])
                self.gate_output[gate] = ng.cast_role(transform(gate_input),
                                                      self.out_axes)

            states['c'] = (states['c'] * self.gate_output['f']
                           + self.gate_output['i'] * self.gate_output['g'])
            states['h'] = self.gate_output['o'] * self.activation(states['c'])
            states['h'] = ng.cast_role(states['h'], self.out_axes)

            output = states['h']

            # append required outputs
            b_k_lists.append(b_k_req)

        return b_k_lists


class Dropout_Modified(Layer):
    """
    Layer for stochastically dropping activations to prevent overfitting.

    Args:
        keep (float):  Number between 0 and 1 that indicates probability of any particular
                       activation being dropped.  Default 0.5.
    """

    def __init__(self, keep=0.5, **kwargs):
        super(Dropout_Modified, self).__init__(**kwargs)
        self.keep = keep
        self.mask = None

    @SubGraph.scope_op_creation
    def __call__(self, in_obj, keep=None, **kwargs):

        if self.mask is None:
            in_axes = in_obj.axes.sample_axes()
            self.mask = ng.persistent_tensor(axes=in_axes).named('mask')
        self.mask = ng.less_equal(
            ng.uniform(
                self.mask,
                low=0.0,
                high=1.0),
            keep)
        return ng.multiply(self.mask, in_obj) * (1. / keep)


class LookupTable(Layer):
    """
    Lookup table layer that often is used as word embedding layerself.

    Args:
        vocab_size (int): the vocabulary size
        embed_dim (int): the size of embedding vector
        init (Initializor): initialization function
        update (bool): if the word vectors get updated through training
        pad_idx (int): by knowing the pad value, the update will make sure always
                       have the vector representing pad value to be 0s.
    """

    def __init__(self, vocab_size, embed_dim, init, update=True, pad_idx=None,
                 **kwargs):
        super(LookupTable, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.init = init
        self.update = update
        self.pad_idx = pad_idx
        self.W = None
        self.W_new = init

    def lut_init(self, axes, pad_word_axis, pad_idx):
        """
        Initialization function for the lut.
        After using the initialization to fill the whole array, set the part that represents
        padding to be 0.
        """
        # init_w = self.init(axes)
        init_w = self.W_new
        if pad_word_axis is 0:
            init_w[pad_idx] = 0
        else:
            init_w[:, pad_idx] = 0
        return init_w

    @SubGraph.scope_op_creation
    def __call__(self, in_obj, **kwargs):
        """
        Arguments:
            in_obj (Tensor): object that provides the lookup indices
        """
        LABELS = {"weight": "weight",
                  "bias": "bias"}

        in_obj = ng.axes_with_order(in_obj,
                                    ng.make_axes([in_obj.axes.recurrent_axis(),
                                                  in_obj.axes.batch_axis()]))
        in_obj = ng.flatten(in_obj)
        in_axes = in_obj.axes

        # label lut_v_axis as shadow axis for initializers ... once #1158 is
        # in, shadow axis will do more than just determine fan in/out for
        # initializers.
        self.lut_v_axis = ng.make_axis(self.vocab_size).named('V')
        self.axes_map = shadow_axes_map([self.lut_v_axis])
        self.lut_v_axis = list(self.axes_map.values())[0]

        self.lut_f_axis = ng.make_axis(self.embed_dim).named('F')

        self.w_axes = ng.make_axes([self.lut_v_axis, self.lut_f_axis])
        self.lut_o_axes = in_axes | ng.make_axes([self.lut_f_axis])
        self.o_axes = ng.make_axes([self.lut_f_axis]) | in_axes[0].axes

        if not self.initialized:
            self.W = ng.variable(
                axes=self.w_axes,
                initial_value=self.lut_init(
                    self.w_axes,
                    self.lut_v_axis,
                    self.pad_idx),
                metadata={
                    "label": LABELS["weight"]},
            ).named('LutW')

        lut_result = ng.lookuptable(
            self.W,
            in_obj,
            self.lut_o_axes,
            update=self.update,
            pad_idx=self.pad_idx)
        return ng.axes_with_order(
            ng.map_roles(ng.unflatten(lut_result), self.axes_map), self.o_axes
        )


def unroll_with_attention(
        cell,
        num_steps,
        H_pr,
        H_hy,
        init_states=None,
        reset_cells=True,
        return_sequence=True,
        reverse_mode=False,
        input_data=None):
    """
    Unroll the cell with attention for num_steps steps.

    Arguments:
    ----------
    cell : provide the cell that has to be unrolled (Eg: MatchLSTMCell_withAttention)
    num_steps: the number of steps needed to unroll
    H_pr : the encoding for the question
    H_hy : the encoding for the passage
    init_states: Either None or a dictionary containing states
    reset_cell: argument which determine if cell has to be reset or not
    reverse_mode: Set to True if unrolling in the opposite direction is desired
    input_data: the ArrayIterator object for training data
                (contains information of length of each sentence)

    """
    recurrent_axis = H_hy.axes.recurrent_axis()

    if init_states is not None:
        states = {k: ng.cast_role(v, out_axes)
                  for (k, v) in init_states.items()}
    else:
        states = init_states

    stepped_inputs = get_steps(H_hy, recurrent_axis, backward=reverse_mode)
    stepped_outputs = []

    for t in range(num_steps):
        with ng.metadata(step=str(t)):
            if t == 0:
                output, states = cell(
                    H_pr, stepped_inputs[t], states, output=None, input_data=input_data)
            else:
                output, states = cell(
                    H_pr, stepped_inputs[t], states, output=output, input_data=input_data)

            stepped_outputs.append(output)

    if reverse_mode:
        if return_sequence:
            stepped_outputs.reverse()

    if return_sequence:
        outputs = ng.stack(stepped_outputs, recurrent_axis, pos=1)
    else:
        outputs = stepped_outputs[-1]

    if not reset_cells:
        update_inits = ng.doall([ng.assign(initial, states[name])
                                 for (name, initial) in states.items()])
        outputs = ng.sequential([update_inits, outputs])

    return outputs
