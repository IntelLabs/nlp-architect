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
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
from neon.layers import BiLSTM
from neon.layers.layer import Layer
from neon.layers.recurrent import interpret_in_shape, get_steps


class DataInput(Layer):
    """
    A layer that is used for inputting data into a network.

    Only supported as the first layer in the network.

    Useful for networks with several input sources that use the
    MergeMulticast merge layer for merging several sources.

    Arguments:
        name (str, optional): Layer name. Defaults to "InputLayer"
    """

    def __init__(self, name=None):
        super(DataInput, self).__init__(name)
        self.owns_output = True

    def __str__(self):
        return "DataInput Layer '%s'" % (self.name)

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(DataInput, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        return self

    def fprop(self, inputs, inference=False):
        """
        Passes input data into the output doing nothing.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        self.outputs = self.inputs = inputs
        return self.outputs

    def _fprop_inference(self, inputs):
        self.fprop(inputs, inference=True)

    def bprop(self, *args):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            *args (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        return None


class TimeDistributedRecurrentOutput(Layer):
    """
    Requires a number of timesteps for which the time distribution will loop

    Options derived from this include:
        TimeDistributedRecurrentLast

    """

    def __init__(self, timesteps, name=None):
        name = name if name else self.classnm
        super(TimeDistributedRecurrentOutput, self).__init__(name)
        self.timesteps = timesteps
        self.owns_output = self.owns_delta = True
        self.x = None

        # Change how the deltas are assigned based on CPU or GPU backend to get
        # around how numpy creates a view of a numpy array when using advanced indexing
        if self.be.backend_name == 'gpu':
            self.assign_deltas_func = self.assign_deltas_gpu
        else:
            self.assign_deltas_func = self.assign_deltas_cpu

    def __str__(self):
        return "TimeDistributedRecurrentOutput choice %s : (%d, %d) inputs, (%d, %d) outputs" % (
            self.name, self.nin, self.nsteps, self.nin, self.nsteps_out)

    def assign_deltas_gpu(self, alpha, error):
        self.deltas_buffer[:, self.out_idxs] = alpha * error

    def assign_deltas_cpu(self, alpha, error):
        self.deltas_buffer._tensor[:, self.out_idxs] = alpha * error._tensor

    def configure(self, in_obj):
        """
        Set shape based parameters of this layer given an input tuple, int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer, Tensor or dataset): object that provides shape
                                                           information for layer

        Returns:
            (tuple): shape of output data
        """
        super(TimeDistributedRecurrentOutput, self).configure(in_obj)
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        self.nsteps_out = int(self.nsteps / self.timesteps)
        self.out_shape = (self.nin, self.nsteps_out)
        return self

    def set_deltas(self, delta_buffers):
        """
        Use pre-allocated (by layer containers) list of buffers for backpropagated error.
        Only set deltas for layers that own their own deltas
        Only allocate space if layer owns its own deltas (e.g., bias and activation work in-place,
        so do not own their deltas).

        Arguments:
            delta_buffers (list): list of pre-allocated tensors (provided by layer container)
        """
        super(TimeDistributedRecurrentOutput, self).set_deltas(delta_buffers)
        self.deltas_buffer = self.deltas
        if self.deltas:
            self.deltas = get_steps(self.deltas_buffer, self.in_shape)
        else:
            self.deltas = []  # for simplifying bprop notation

    def init_buffers(self, inputs):
        """
        Initialize buffers for recurrent internal units and outputs.
        Buffers are initialized as 2D tensors with second dimension being steps * batch_size
        A list of views are created on the buffer for easy manipulation of data
        related to a certain time step

        Arguments:
            inputs (Tensor): input data as 2D tensor. The dimension is
                             (input_size, sequence_length * batch_size)

        """
        if self.x is None or self.x is not inputs:
            self.x = inputs
            self.xs = get_steps(inputs, self.in_shape)


class TimeDistributedRecurrentLast(TimeDistributedRecurrentOutput):
    """
    A layer that only keeps the recurrent layer output at the last time step
    of each self.timesteps.
    """

    def configure(self, in_obj):
        """
        Precompute the indicies of the time distributed outputs in the input
        array.
        """
        super(TimeDistributedRecurrentLast, self).configure(in_obj)
        self.out_idxs = [list(range(self.be.bsz * (i * self.timesteps - 1),
                                    self.be.bsz * (i * self.timesteps)))
                         for i in range(1, self.nsteps_out + 1)]
        self.out_idxs = np.reshape(self.out_idxs, -1)
        return self

    def fprop(self, inputs, inference=False):
        """
        Takes each self.timesteps element from the input along the sequence dimension

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        self.init_buffers(inputs)
        self.outputs[:] = self.x[:, self.out_idxs]
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.deltas:
            # RNN/LSTM layers don't allocate new hidden units delta buffers and they overwrite it
            # while doing bprop. So, init with zeros here.
            self.deltas_buffer.fill(0)
            self.assign_deltas_func(alpha, error)

        return self.deltas_buffer


class TimeDistBiLSTM(BiLSTM):
    """
    A Bi-directional LSTM layer that supports time step output of the LSTM layer.
    """
    def __init__(self, output_size, init, init_inner=None, activation=None,
                 gate_activation=None, reset_cells=False, reset_freq=0,
                 split_inputs=False, name=None):
        super(TimeDistBiLSTM, self).__init__(output_size, init, init_inner=init_inner,
                                             activation=activation,
                                             gate_activation=gate_activation,
                                             reset_cells=reset_cells,
                                             split_inputs=split_inputs, name=name)
        self.reset_freq = reset_freq

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (list): list of Tensors with one such tensor for each time
                           step of model unrolling.
            inference (bool, optional): Set to true if you are running
                                        inference (only care about forward
                                        propagation without associated backward
                                        propagation).  Default is False.

        Returns:
            Tensor: LSTM output for each model time step
        """
        self.init_buffers(inputs)  # calls the BiRNN init_buffers() code

        if self.reset_cells:
            self.h_f[-1][:] = 0
            self.c_f[-1][:] = 0
            self.h_b[0][:] = 0
            self.c_b[0][:] = 0

        params_f = (self.h_f, self.h_prev, self.xs_f, self.ifog_f, self.ifo_f,
                    self.i_f, self.f_f, self.o_f, self.g_f, self.c_f, self.c_prev, self.c_act_f)
        params_b = (self.h_b, self.h_next, self.xs_b, self.ifog_b, self.ifo_b,
                    self.i_b, self.f_b, self.o_b, self.g_b, self.c_b, self.c_next, self.c_act_b)

        self.be.compound_dot(self.W_input_f, self.x_f, self.ifog_buffer_f)
        self.be.compound_dot(self.W_input_b, self.x_b, self.ifog_buffer_b)

        for idx, (h, h_prev, xs, ifog, ifo, i, f, o, g, c, c_prev, c_act) \
                in enumerate(zip(*params_f)):
            if self.reset_freq > 0 and idx % self.reset_freq == 0:
                h_prev = self.be.zeros(h_prev.shape)
                c_prev = self.be.zeros(c_prev.shape)

            self.be.compound_dot(self.W_input_f, xs, ifog)
            self.be.compound_dot(self.W_recur_f, h_prev, ifog, beta=1.0)
            ifog[:] = ifog + self.b_f

            ifo[:] = self.gate_activation(ifo)
            g[:] = self.activation(g)

            c[:] = f * c_prev + i * g
            c_act[:] = self.activation(c)
            h[:] = o * c_act

        for idx, (h, h_next, xs, ifog, ifo, i, f, o, g, c, c_next, c_act) in \
                enumerate(reversed(list(zip(*params_b)))):
            if self.reset_freq > 0 and idx % self.reset_freq == 0:
                h_next = self.be.zeros(h_next.shape)
                c_next = self.be.zeros(c_next.shape)

            self.be.compound_dot(self.W_recur_b, h_next, ifog)
            self.be.compound_dot(self.W_input_b, xs, ifog, beta=1.0)
            ifog[:] = ifog + self.b_b

            ifo[:] = self.gate_activation(ifo)
            g[:] = self.activation(g)
            c[:] = f * c_next + i * g
            c_act[:] = self.activation(c)
            h[:] = o * c_act

        return self.h_buffer

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Backpropagation of errors, output delta for previous layer, and
        calculate the update on model params

        Arguments:
            error (list[Tensor]): error tensors for each time step
                                  of unrolling
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: Backpropagated errors for each time step of model unrolling
        """
        self.dW[:] = 0

        if self.in_deltas_f is None:
            self.in_deltas_f = get_steps(error[:self.o_shape[0]], self.o_shape)
            self.prev_in_deltas = self.in_deltas_f[-1:] + self.in_deltas_f[:-1]
            self.ifog_delta_last_steps = self.ifog_delta_buffer[:, self.be.bsz:]
            self.h_first_steps = self.h_buffer_f[:, :-self.be.bsz]
            # h_delta[5] * h[4] + h_delta[4] * h[3] + ... + h_delta[1] * h[0]

        if self.in_deltas_b is None:
            self.in_deltas_b = get_steps(error[self.o_shape[0]:], self.o_shape)
            self.next_in_deltas = self.in_deltas_b[1:] + self.in_deltas_b[:1]
            self.ifog_delta_first_steps = self.ifog_delta_buffer[:, :-self.be.bsz]
            self.h_last_steps = self.h_buffer_b[:, self.be.bsz:]
            # h_delta[0] * h[1] + h_delta[1] * h[2] + ... + h_delta[4] * h[5]

        params_f = (self.in_deltas_f, self.prev_in_deltas,
                    self.i_f, self.f_f, self.o_f, self.g_f,
                    self.ifog_delta, self.i_delta, self.f_delta, self.o_delta, self.g_delta,
                    self.c_delta, self.c_delta_prev, self.c_prev_bprop, self.c_act_f)

        params_b = (self.in_deltas_b, self.next_in_deltas,
                    self.i_b, self.f_b, self.o_b, self.g_b,
                    self.ifog_delta, self.i_delta, self.f_delta, self.o_delta, self.g_delta,
                    self.c_delta, self.c_delta_next, self.c_next_bprop, self.c_act_b)

        # bprop for forward direction connections . Error flow from right to left
        self.c_delta_buffer[:] = 0
        self.ifog_delta_buffer[:] = 0
        self.ifog_delta_f = None
        self.ifog_delta_b = None
        for idx, (in_deltas, prev_in_deltas,
                  i, f, o, g,
                  ifog_delta, i_delta, f_delta, o_delta, g_delta,
                  c_delta, c_delta_prev, c_prev, c_act) \
                in enumerate(reversed(list(zip(*params_f)))):

            # current cell delta
            c_delta[:] = c_delta + \
                         self.activation.bprop(c_act) * (o * in_deltas)
            i_delta[:] = self.gate_activation.bprop(i) * c_delta * g
            f_delta[:] = self.gate_activation.bprop(f) * c_delta * c_prev
            o_delta[:] = self.gate_activation.bprop(o) * in_deltas * c_act
            g_delta[:] = self.activation.bprop(g) * c_delta * i

            # bprop the errors to prev_in_delta and c_delta_prev
            self.be.compound_dot(
                    self.W_recur_f.T, ifog_delta, prev_in_deltas, beta=1.0)
            if c_delta_prev is not None:
                c_delta_prev[:] = c_delta * f

        # Weight deltas and accumulate
        self.be.compound_dot(
                self.ifog_delta_last_steps, self.h_first_steps.T, self.dW_recur_f)
        self.be.compound_dot(
                self.ifog_delta_buffer, self.x_f.T, self.dW_input_f)
        self.db_f[:] = self.be.sum(self.ifog_delta_buffer, axis=1)
        # out deltas to input units
        if self.out_deltas_buffer:
            self.be.compound_dot(
                    self.W_input_f.T, self.ifog_delta_buffer,
                    self.out_deltas_buffer_f_v,
                    alpha=alpha, beta=beta)

        # bprop for backward direction connections. Error flow from left to right
        self.c_delta_buffer[:] = 0
        self.ifog_delta_buffer[:] = 0
        for idx, (in_deltas, next_in_deltas,
                  i, f, o, g,
                  ifog_delta, i_delta, f_delta, o_delta, g_delta,
                  c_delta, c_delta_next, c_next, c_act) \
                in enumerate(zip(*params_b)):

            # current cell delta
            c_delta[:] = c_delta[:] + \
                         self.activation.bprop(c_act) * (o * in_deltas)
            i_delta[:] = self.gate_activation.bprop(i) * c_delta * g
            f_delta[:] = self.gate_activation.bprop(f) * c_delta * c_next
            o_delta[:] = self.gate_activation.bprop(o) * in_deltas * c_act
            g_delta[:] = self.activation.bprop(g) * c_delta * i

            # bprop the errors to next_in_delta and c_next_delta
            self.be.compound_dot(
                    self.W_recur_b.T, ifog_delta, next_in_deltas, beta=1.0)
            if c_delta_next is not None:
                c_delta_next[:] = c_delta * f

        # Weight deltas and accumulate
        self.be.compound_dot(
                self.ifog_delta_first_steps, self.h_last_steps.T, self.dW_recur_b)
        self.be.compound_dot(
                self.ifog_delta_buffer, self.x_b.T, self.dW_input_b)
        self.db_b[:] = self.be.sum(self.ifog_delta_buffer, axis=1)
        # out deltas to input units. bprop to the same inputs if
        # split_inputs=False
        if self.out_deltas_buffer:
            self.be.compound_dot(self.W_input_b.T, self.ifog_delta_buffer,
                                 self.out_deltas_buffer_b_v,
                                 alpha=alpha,
                                 beta=beta if self.inputs else 1.0)

        return self.out_deltas_buffer
