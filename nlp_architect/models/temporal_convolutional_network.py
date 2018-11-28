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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
# pylint: disable=no-name-in-module
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import nn_ops
from tensorflow.python.layers.convolutional import Conv1D
from tensorflow.python.keras import layers as keras_layers


class _ConvWeightNorm(keras_layers.Conv1D, base.Layer):
    """
    Convolution base class that uses weight norm
    """
    def __init__(self, *args,
                 **kwargs):
        super(_ConvWeightNorm, self).__init__(*args,
                                              **kwargs)
        self.kernel_v = None
        self.kernel_g = None
        self.kernel = None
        self.bias = None
        self._convolution_op = None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        # pylint: disable=no-member
        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        # pylint: disable=no-member
        input_dim = input_shape[channel_axis].value
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        # The variables defined below are specific to the weight normed conv class
        self.kernel_v = self.add_variable(name='kernel_v',
                                          shape=kernel_shape,
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
        self.kernel_g = self.add_variable(name='kernel_g', shape=[], trainable=True,
                                          dtype=self.dtype)
        self.kernel = self.kernel_g * tf.nn.l2_normalize(self.kernel_v)

        if self.use_bias:
            self.bias = self.add_variable(name='bias',
                                          shape=(self.filters,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
        else:
            self.bias = None
        self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                         axes={channel_axis: input_dim})
        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=self.kernel.get_shape(),
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=utils.convert_data_format(self.data_format,
                                                  self.rank + 2))
        self.built = True


# re-orient the Conv1D class to point to the weight norm version of conv base class
Conv1D.__bases__ = (_ConvWeightNorm,)


class TCN:
    """
    This class defines core TCN architecture.
    This is only the base class, training strategy is not implemented.
    """
    def __init__(self, max_len, n_features_in, hidden_sizes, kernel_size=7, dropout=0.2):
        """
        To use this class,
            1. Inherit this class
            2. Define the training losses in build_train_graph()
            3. Define the training strategy in run()
            4. After the inherited class object is initialized,
               call build_train_graph followed by run

        Args:
            max_len: Maximum length of sequence
            n_features_in: Number of input features (dimensions)
            hidden_sizes: Number of hidden sizes in each layer of TCN (same for all layers)
            kernel_size: Kernel size of convolution filter (same for all layers)
            dropout: Dropout, fraction of activations to drop
        """
        self.max_len = max_len
        self.n_features_in = n_features_in
        self.hidden_sizes = hidden_sizes
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_hidden_layers = len(self.hidden_sizes)
        receptive_field_len = self.calculate_receptive_field()
        if receptive_field_len < self.max_len:
            print("Warning! receptive field of the TCN: "
                  "%d is less than the input sequence length: %d."
                  % (receptive_field_len, self.max_len))
        else:
            print("Receptive field of the TCN: %d, input sequence length: %d."
                  % (receptive_field_len, self.max_len))
        self.layer_activations = []

        # toggle this for train/inference mode
        self.training_mode = tf.placeholder(tf.bool, name='training_mode')

        self.sequence_output = None

    def calculate_receptive_field(self):
        """

        Returns:

        """
        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.n_hidden_layers - 1)

    def build_network_graph(self, x, last_timepoint=False):
        """
        Given the input placeholder x, build the entire TCN graph
        Args:
            x: Input placeholder
            last_timepoint: Whether or not to select only the last timepoint to output

        Returns:
            output of the TCN
        """
        # loop and define multiple residual blocks
        with tf.variable_scope("tcn"):
            for i in range(self.n_hidden_layers):
                dilation_size = 2 ** i
                in_channels = self.n_features_in if i == 0 else self.hidden_sizes[i - 1]
                out_channels = self.hidden_sizes[i]
                with tf.variable_scope("residual_block_" + str(i)):
                    x = self._residual_block(x, in_channels, out_channels, dilation_size,
                                             (self.kernel_size - 1) * dilation_size)
                    x = tf.nn.relu(x)
                self.layer_activations.append(x)
            self.sequence_output = x

            # get outputs
            if not last_timepoint:
                prediction = self.sequence_output
            else:
                # last time point size (batch_size, hidden_sizes_encoder)
                width = self.sequence_output.shape[1].value
                lt = tf.squeeze(tf.slice(self.sequence_output, [0, width - 1, 0],
                                         [-1, 1, -1]), axis=1)
                prediction = \
                    tf.layers.Dense(1, kernel_initializer=tf.initializers.random_normal(0, 0.01),
                                    bias_initializer=tf.initializers.random_normal(0, 0.01))(lt)

        return prediction

    def _residual_block(self, x, in_channels, out_channels, dilation, padding):
        """
        Defines the residual block
        Args:
            x: Input tensor to residual block
            in_channels: Number of input features (dimensions)
            out_channels: Number of output features (dimensions)
            dilation: Dilation rate
            padding: Padding value

        Returns:
            Output of residual path
        """
        xin = x
        # define two temporal blocks
        for i in range(2):
            with tf.variable_scope("temporal_block_" + str(i)):
                x = self._temporal_block(x, out_channels, dilation, padding)

        # sidepath
        if in_channels != out_channels:
            x_side = tf.layers.Conv1D(filters=out_channels, kernel_size=1, padding='same',
                                      strides=1, activation=None, dilation_rate=1,
                                      kernel_initializer=tf.initializers.random_normal(0, 0.01),
                                      bias_initializer=tf.initializers.random_normal(0, 0.01))(xin)
        else:
            x_side = xin

        # combine both
        return tf.add(x, x_side)

    def _temporal_block(self, x, out_channels, dilation, padding):
        """
        Defines the temporal block, which is a dilated causual conv layer,
        followed by relu and dropout
        Args:
            x: Input to temporal block
            out_channels: Number of conv filters
            dilation: dilation rate
            padding: padding value

        Returns:
            Tensor output of temporal block
        """
        # conv layer
        x = self._dilated_causal_conv(x, out_channels, dilation, padding)

        x = tf.nn.relu(x)

        # dropout
        batch_size = tf.shape(x)[0]
        x = tf.layers.dropout(x, rate=self.dropout, noise_shape=[batch_size, 1, out_channels],
                              training=self.training_mode)

        return x

    # define model
    def _dilated_causal_conv(self, x, n_filters, dilation, padding):
        """
        Defines dilated causal convolution
        Args:
            x: Input activation
            n_filters: Number of convolution filters
            dilation: Dilation rate
            padding: padding value

        Returns:
            Tensor output of convolution
        """
        input_width = x.shape[1].value
        with tf.variable_scope("dilated_causal_conv"):
            # define dilated convolution layer with left side padding
            x = tf.pad(x, tf.constant([[0, 0], [padding, 0], [0, 0]]), 'CONSTANT')
            x = Conv1D(filters=n_filters, kernel_size=self.kernel_size, padding='valid', strides=1,
                       activation=None, dilation_rate=dilation,
                       kernel_initializer=tf.initializers.random_normal(0, 0.01),
                       bias_initializer=tf.initializers.random_normal(0, 0.01))(x)

        assert x.shape[1].value == input_width

        return x

    def build_train_graph(self, *args, **kwargs):
        """
        Placeholder for defining training losses and metrics
        """
        raise NotImplementedError("Error! losses for training must be defined")

    def run(self, *args, **kwargs):
        """
        Placeholder for defining training strategy
        """
        raise NotImplementedError("Error! training routine must be defined")


class CommonLayers:
    """
    Class that contains the common layers for language modeling -
            word embeddings and projection layer
    """
    def __init__(self):
        """
        Initialize class
        """
        self.word_embeddings_tf = None
        self.num_words = None
        self.n_features_in = None

    def define_input_layer(self, input_placeholder_tokens, word_embeddings,
                           embeddings_trainable=True):
        """
        Define the input word embedding layer
        Args:
            input_placeholder_tokens: tf.placeholder, input to the model
            word_embeddings: numpy array (optional), to initialize the embeddings with
            embeddings_trainable: boolean, whether or not to train the embedding table

        Returns:
            Embeddings corresponding to the data in input placeholder
        """
        with tf.device('/cpu:0'):
            with tf.variable_scope("embedding_layer", reuse=False):
                if word_embeddings is None:
                    initializer = tf.initializers.random_normal(0, 0.01)
                else:
                    initializer = tf.constant_initializer(word_embeddings)
                self.word_embeddings_tf = tf.get_variable("embedding_table",
                                                          shape=[self.num_words,
                                                                 self.n_features_in],
                                                          initializer=initializer,
                                                          trainable=embeddings_trainable)

                input_embeddings = tf.nn.embedding_lookup(self.word_embeddings_tf,
                                                          input_placeholder_tokens)
        return input_embeddings

    def define_projection_layer(self, prediction, tied_weights=True):
        """
        Define the output word embedding layer
        Args:
            prediction: tf.tensor, the prediction from the model
            tied_weights: boolean, whether or not to tie weights from the input embedding layer

        Returns:
            Probability distribution over vocabulary
        """
        with tf.device('/cpu:0'):
            if tied_weights:
                # tie projection layer and embedding layer
                with tf.variable_scope("embedding_layer", reuse=tf.AUTO_REUSE):
                    softmax_w = tf.matrix_transpose(self.word_embeddings_tf)
                    softmax_b = tf.get_variable('softmax_b', [self.num_words])
                    _, l, k = prediction.shape.as_list()
                    prediction_reshaped = tf.reshape(prediction, [-1, k])
                    mult_out = tf.nn.bias_add(tf.matmul(prediction_reshaped, softmax_w), softmax_b)
                    projection_out = tf.reshape(mult_out, [-1, l, self.num_words])
            else:
                with tf.variable_scope("projection_layer", reuse=False):
                    projection_out = tf.layers.Dense(self.num_words)(prediction)
        return projection_out
