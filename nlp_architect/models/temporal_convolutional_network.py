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
from tensorflow.python.keras.layers import Wrapper
from tensorflow.python.layers.convolutional import Conv1D
from tensorflow.python.ops import variable_scope
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.eager import context
from tensorflow.python.ops import nn_impl
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops


# ***NOTE***: The WeightNorm Class is copied from this PR:
# https://github.com/tensorflow/tensorflow/issues/14070
# Once this becomes part of the official TF release, it will be removed
class WeightNorm(Wrapper):
    """ This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction. This speeds up convergence by improving the
    conditioning of the optimization problem.

    Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
    Tim Salimans, Diederik P. Kingma (2016)

    WeightNorm wrapper works for keras and tf layers.

    ```python
      net = WeightNorm(tf.keras.layers.Conv2D(2, 2, activation='relu'),
             input_shape=(32, 32, 3), data_init=True)(x)
      net = WeightNorm(tf.keras.layers.Conv2D(16, 5, activation='relu'),
                       data_init=True)
      net = WeightNorm(tf.keras.layers.Dense(120, activation='relu'),
                       data_init=True)(net)
      net = WeightNorm(tf.keras.layers.Dense(n_classes),
                       data_init=True)(net)
    ```

    Arguments:
      layer: a layer instance.
      data_init: If `True` use data dependent variable initialization

    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights
      NotImplementedError: If `data_init` is True and running graph execution
    """
    def __init__(self, layer, data_init=False, **kwargs):
        if not isinstance(layer, Layer):
            raise ValueError(
                'Please initialize `WeightNorm` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))

        if not context.executing_eagerly() and data_init:
            raise NotImplementedError(
                'Data dependent variable initialization is not available for '
                'graph execution')

        self.initialized = True
        if data_init:
            self.initialized = False

        self.layer_depth = None
        self.norm_axes = None
        super(WeightNorm, self).__init__(layer, **kwargs)
        self._track_checkpointable(layer, name='layer')

    def _compute_weights(self):
        """Generate weights by combining the direction of weight vector
         with it's norm """
        with variable_scope.variable_scope('compute_weights'):
            self.layer.kernel = nn_impl.l2_normalize(
                self.layer.v, axis=self.norm_axes) * self.layer.g

    def _init_norm(self, weights):
        """Set the norm of the weight vector"""
        from tensorflow.python.ops.linalg_ops import norm
        with variable_scope.variable_scope('init_norm'):
            # pylint: disable=no-member
            flat = array_ops.reshape(weights, [-1, self.layer_depth])
            # pylint: disable=no-member
            return array_ops.reshape(norm(flat, axis=0), (self.layer_depth,))

    def _data_dep_init(self, inputs):
        """Data dependent initialization for eager execution"""
        from tensorflow.python.ops.nn import moments
        from tensorflow.python.ops.math_ops import sqrt

        with variable_scope.variable_scope('data_dep_init'):
            # Generate data dependent init values
            activation = self.layer.activation
            self.layer.activation = None
            x_init = self.layer.call(inputs)
            m_init, v_init = moments(x_init, self.norm_axes)
            scale_init = 1. / sqrt(v_init + 1e-10)

        # Assign data dependent init values
        self.layer.g = self.layer.g * scale_init
        self.layer.bias = (-1 * m_init * scale_init)
        self.layer.activation = activation
        self.initialized = True

    # pylint: disable=signature-differs
    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        self.input_spec = InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = False

            if not hasattr(self.layer, 'kernel'):
                raise ValueError(
                    '`WeightNorm` must wrap a layer that'
                    ' contains a `kernel` for weights'
                )

            # The kernel's filter or unit dimension is -1
            self.layer_depth = int(self.layer.kernel.shape[-1])
            self.norm_axes = list(range(self.layer.kernel.shape.ndims - 1))

            self.layer.v = self.layer.kernel
            self.layer.g = self.layer.add_variable(
                name="g",
                shape=(self.layer_depth,),
                initializer=initializers.get('ones'),
                dtype=self.layer.kernel.dtype,
                trainable=True)

            with ops.control_dependencies([self.layer.g.assign(
                    self._init_norm(self.layer.v))]):
                self._compute_weights()

            self.layer.built = True

        super(WeightNorm, self).build()
        self.built = True

    # pylint: disable=arguments-differ
    def call(self, inputs):
        """Call `Layer`"""
        if context.executing_eagerly():
            if not self.initialized:
                self._data_dep_init(inputs)
            self._compute_weights()  # Recompute weights for each forward pass

        output = self.layer.call(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())


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
            x = WeightNorm(Conv1D(filters=n_filters, kernel_size=self.kernel_size, padding='valid',
                                  strides=1, activation=None, dilation_rate=dilation,
                                  kernel_initializer=tf.initializers.random_normal(0, 0.01),
                                  bias_initializer=tf.initializers.random_normal(0, 0.01)))(x)

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
