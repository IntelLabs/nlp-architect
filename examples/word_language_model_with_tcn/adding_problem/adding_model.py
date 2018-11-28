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
import os

import tensorflow as tf

from nlp_architect.models.temporal_convolutional_network import TCN


class TCNForAdding(TCN):
    """
    Main class that defines training graph and defines training run method for the adding problem
    """
    def __init__(self, *args, **kwargs):
        super(TCNForAdding, self).__init__(*args, **kwargs)
        self.input_placeholder = None
        self.label_placeholder = None
        self.prediction = None
        self.training_loss = None
        self.merged_summary_op_train = None
        self.merged_summary_op_val = None
        self.training_update_step = None

    # pylint: disable = arguments-differ
    def run(self, data_loader, num_iterations=1000, log_interval=100, result_dir="./"):
        """
        Runs training
        Args:
            data_loader: iterator, Data loader for adding problem
            num_iterations: int, number of iterations to run
            log_interval: int, number of iterations after which to run validation and log
            result_dir: str, path to results directory

        Returns:
            None
        """
        summary_writer = tf.summary.FileWriter(os.path.join(result_dir, "tfboard"),
                                               tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=None)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        for i in range(num_iterations):

            x_data, y_data = next(data_loader)

            feed_dict = {self.input_placeholder: x_data, self.label_placeholder: y_data,
                         self.training_mode: True}
            _, summary_train, total_loss_i = sess.run([self.training_update_step,
                                                       self.merged_summary_op_train,
                                                       self.training_loss], feed_dict=feed_dict)

            summary_writer.add_summary(summary_train, i)

            if i % log_interval == 0:
                print("Step {}: Total: {}".format(i, total_loss_i))
                saver.save(sess, result_dir, global_step=i)

                feed_dict = {self.input_placeholder: data_loader.test[0],
                             self.label_placeholder: data_loader.test[1],
                             self.training_mode: False}
                val_loss, summary_val = sess.run([self.training_loss, self.merged_summary_op_val],
                                                 feed_dict=feed_dict)

                summary_writer.add_summary(summary_val, i)

                print("Validation loss: {}".format(val_loss))

    # pylint: disable = arguments-differ
    def build_train_graph(self, lr, max_gradient_norm=None):
        """
        Method that builds the graph for training
        Args:
            lr: float, learning rate
            max_gradient_norm: float, maximum gradient norm value for clipping

        Returns:
            None
        """
        with tf.variable_scope("input", reuse=True):
            self.input_placeholder = tf.placeholder(tf.float32,
                                                    [None, self.max_len, self.n_features_in],
                                                    name='input')
            self.label_placeholder = tf.placeholder(tf.float32, [None, 1], name='labels')

        self.prediction = self.build_network_graph(self.input_placeholder, last_timepoint=True)

        with tf.variable_scope("training"):
            self.training_loss = tf.losses.mean_squared_error(self.label_placeholder,
                                                              self.prediction)

            summary_ops_train = [tf.summary.scalar("Training Loss", self.training_loss)]
            self.merged_summary_op_train = tf.summary.merge(summary_ops_train)

            summary_ops_val = [tf.summary.scalar("Validation Loss", self.training_loss)]
            self.merged_summary_op_val = tf.summary.merge(summary_ops_val)

            # Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(self.training_loss, params)
            if max_gradient_norm is not None:
                clipped_gradients = [tf.clip_by_norm(t, max_gradient_norm) for t in gradients]
            else:
                clipped_gradients = gradients

            # Optimization
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            optimizer = tf.train.AdamOptimizer(lr)
            with tf.control_dependencies(update_ops):
                self.training_update_step = optimizer.apply_gradients(zip(clipped_gradients,
                                                                          params))
