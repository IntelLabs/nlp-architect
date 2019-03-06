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
# ****************************************************************************
import tensorflow as tf


class MostCommonWordSense(object):
    def __init__(self, epochs, batch_size, callback_args=None):
        self.optimizer = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.loss = 'mean_squared_error'
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.callback_args = callback_args

    def build(self, input_dim):
        # setup model layers
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(100, activation='relu', input_dim=input_dim))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(2, activation='softmax'))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        self.model = model

    def fit(self, train_set):
        self.model.fit(train_set['X'], train_set['y'], epochs=self.epochs,
                       batch_size=self.batch_size)

    def save(self, save_path):
        self.model.save(save_path)

    def load(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def eval(self, valid_set):
        eval_rate = self.model.evaluate(valid_set['X'], valid_set['y'], batch_size=self.batch_size)
        return eval_rate

    def get_outputs(self, valid_set):
        return self.model.predict(valid_set)
