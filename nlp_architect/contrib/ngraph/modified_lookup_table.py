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
import ngraph as ng
from ngraph.frontends.neon.axis import shadow_axes_map
from ngraph.frontends.neon import Layer
from ngraph.frontends.neon import GaussianInit
from ngraph.frontends.neon.graph import SubGraph

LABELS = {"weight": "weight",
          "bias": "bias"}


class ModifiedLookupTable(Layer):
    """
    Lookup table layer that often is used as word embedding layer.

    Modified from the default LookupTable implementation to support multiple axis lookups.

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
        super(ModifiedLookupTable, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.init = init
        self.update = update
        self.pad_idx = pad_idx
        self.W = None

    def lut_init(self, axes, pad_word_axis, pad_idx):
        """
        Initialization function for the lut.
        After using the initialization to fill the whole array, set the part that represents
        padding to be 0.
        """
        init_w = self.init(axes)
        if axes.index(pad_word_axis) is 0:
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
        return ng.map_roles(ng.unflatten(lut_result), self.axes_map)
