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
# pylint: disable=all

from __future__ import division
from __future__ import print_function
import ngraph as ng
import numpy as np
from ngraph.testing.random import RandomTensorGenerator


# Contains functions to initialze LSTM cells


def weight_initializer(request):
    if request.param == "random":
        return lambda w_axes: rng.normal(0, 1, w_axes)  # noqa: F821
    elif request.param == "ones":
        return lambda w_axes: np.zeros(w_axes.lengths)


def make_weights(
        input_placeholder,
        hidden_size,
        weight_initializer,
        bias_initializer,
        init_state=False):
    gates = ['i', 'f', 'o', 'g']

    # input axis + any extra axes of length 1
    in_feature_axes = tuple(input_placeholder.axes)[:-2]
    out_feature_axes = ng.make_axes([ng.make_axis(hidden_size)])
    batch_axis = input_placeholder.axes.batch_axis()
    hidden_axis = ng.make_axis(hidden_size)

    w_in_axes = ng.make_axes(hidden_axis) + in_feature_axes
    w_rec_axes = ng.make_axes(hidden_axis) + out_feature_axes

    W_in = {gate: weight_initializer(w_in_axes) for gate in gates}
    W_rec = {gate: weight_initializer(w_rec_axes) for gate in gates}
    b = {gate: bias_initializer(hidden_axis) for gate in gates}

    if init_state is True:
        ax_s = ng.make_axes([hidden_axis, batch_axis])
        init_state = {name: ng.placeholder(ax_s) for name in ['h', 'c']}
        init_state_value = {
            name: rng.uniform(-1, 1, ax_s) for name in ['h', 'c']}  # noqa: F821
    else:
        init_state = None
        init_state_value = None

    return W_in, W_rec, b, init_state, init_state_value


def make_placeholder(input_size, sequence_length, batch_size, extra_axes=0):

    input_axis = ng.make_axis(name='features')
    recurrent_axis = ng.make_axis(name='REC_REP')
    batch_axis = ng.make_axis(name='N')

    input_axes = ng.make_axes([input_axis, recurrent_axis, batch_axis])
    input_axes.set_shape((input_size, sequence_length, batch_size))
    input_axes = ng.make_axes([ng.make_axis(length=1, name='features_' + str(i))
                               for i in range(extra_axes)]) + input_axes

    input_placeholder = ng.placeholder(input_axes)
    rng = RandomTensorGenerator()
    input_value = rng.uniform(-0.01, 0.01, input_axes)

    return input_placeholder, input_value
