import tensorflow as tf
import os
from examples.tcn.utils import Conv1DWeightNorm


class TCN:
    def __init__(self, max_len, n_features_in, hidden_sizes, kernel_size=7, dropout=0.2, last_timepoint=False):
        self.max_len = max_len
        self.n_features_in = n_features_in
        self.hidden_sizes = hidden_sizes
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_hidden_layers = len(self.hidden_sizes)
        self.last_timepoint = last_timepoint

        ## define input placeholders
        with tf.variable_scope("input"):
            self.training_mode = tf.placeholder(tf.bool, name='training_mode')

    def _get_predictions(self):
        # get outputs
        if not self.last_timepoint:
            self.prediction = self.sequence_output
        else:
            width = self.sequence_output.shape[1].value
            last_tp = tf.squeeze(tf.slice(self.sequence_output, [0, width - 1, 0], [-1, 1, -1]), axis=1)  # last time point size (batch_size, hidden_sizes_encoder)
            self.prediction = tf.layers.Dense(1, kernel_initializer=tf.initializers.random_normal(0, 0.01), bias_initializer=tf.initializers.random_normal(0, 0.01))(last_tp)

    def _build_network_graph(self, x):
        # loop and define multiple residual blocks
        with tf.variable_scope("tcn"):
            for i in range(self.n_hidden_layers):
                dilation_size = 2 ** i
                in_channels = self.n_features_in if i == 0 else self.hidden_sizes[i - 1]
                out_channels = self.hidden_sizes[i]
                with tf.variable_scope("residual_block_" + str(i)):
                    x = self._residual_block(x, in_channels, out_channels, dilation_size, (self.kernel_size - 1) * dilation_size)
                    x = tf.nn.relu(x)
        self.sequence_output = x

    ## define residual block
    def _residual_block(self, x, in_channels, out_channels, dilation, padding):

        x_in = x
        # define two temporal blocks
        for i in range(2):
            with tf.variable_scope("temporal_block_" + str(i)):
                x = self._temporal_block(x, out_channels, dilation, padding)

        # sidepath
        if in_channels != out_channels:
            x_side = tf.layers.Conv1D(filters=out_channels, kernel_size=1, padding='same', strides=1, activation=None, dilation_rate=1, kernel_initializer=tf.initializers.random_normal(0, 0.01), bias_initializer=tf.initializers.random_normal(0, 0.01))(x_in)
        else:
            x_side = x_in

        # combine both
        return tf.add(x, x_side)

    ## define temporal block
    def _temporal_block(self, x, out_channels, dilation, padding):
        # conv layer
        x = self._dilated_causal_conv(x, out_channels, dilation, padding)

        x = tf.nn.relu(x)

        # dropout
        x = tf.layers.dropout(x, rate=self.dropout, noise_shape=[None, 1, out_channels], training=self.training_mode)

        return x

    # define model
    def _dilated_causal_conv(self, x, n_filters, dilation, padding):
        input_width = x.shape[1].value
        with tf.variable_scope("dilated_causal_conv"):
            # define dilated convolution layer
            x = tf.pad(x, tf.constant([[0, 0], [padding, padding], [0, 0]]), 'CONSTANT') # both side padding
            x = Conv1DWeightNorm(filters=n_filters, kernel_size=self.kernel_size, padding='valid', strides=1, activation=None, dilation_rate=dilation, kernel_initializer=tf.initializers.random_normal(0, 0.01), bias_initializer=tf.initializers.random_normal(0, 0.01))(x)
            x = tf.slice(x, [0, 0, 0], [-1, input_width, -1])

        assert x.shape[1].value == input_width

        return x

    def build_train_graph(self, *args, **kwargs):
        raise NotImplementedError("Error! losses for training must be defined")

    def run(self, *args, **kwargs):
        raise NotImplementedError("Error! training routine must be defined")

    def set_up_callbacks(self, result_dir):
        self.summary_writer = tf.summary.FileWriter(os.path.join(result_dir, "tfboard"), tf.get_default_graph())
        self.saver = tf.train.Saver(max_to_keep=None)
