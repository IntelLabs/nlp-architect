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

from __future__ import division, print_function, unicode_literals, absolute_import

import pickle
import tempfile

import tensorflow as tf
from tensorflow import keras


class SequenceChunker(object):
    """
    A sequence Chunker model written in Tensorflow (and Keras) based on the
    paper 'Deep multi-task learning with low level tasks supervised at lower layers'.
    The model has 3 Bi-LSTM layers and outputs POS and Chunk tags.

    Args:
        use_gpu (bool, optional): use GPU based model (CUDNNA cells)
    """

    def __init__(self, use_gpu=False):
        self.vocabulary_size = None
        self.num_pos_labels = None
        self.num_chunk_labels = None
        self.feature_size = None
        self.dropout = None
        self.optimizer = None
        self.model = None
        self.use_gpu = use_gpu

    def build(self,
              vocabulary_size,
              num_pos_labels,
              num_chunk_labels,
              feature_size=100,
              dropout=0.5,
              optimizer=None):
        """
        Build a chunker/POS model

        Args:
            vocabulary_size (int): the size of the input vocabulary
            num_pos_labels (int): the size of of POS labels
            num_chunk_labels (int): the sie of chunk labels
            feature_size (int, optional): feature size - determines the embedding/LSTM layer \
                hidden state size
            dropout (float, optional): dropout rate
            optimizer (tensorflow.python.training.optimizer.Optimizer, optional): optimizer, if \
                None will use default SGD (paper setup)
        """
        self.vocabulary_size = vocabulary_size
        self.num_pos_labels = num_pos_labels
        self.num_chunk_labels = num_chunk_labels
        self.feature_size = feature_size
        self.dropout = dropout
        embedding_layer = self._embedding_layer()
        word_input = keras.layers.Input(shape=(None,))
        word_embedding = embedding_layer(word_input)
        rnn_layer_1 = keras.layers.Bidirectional(self._rnn_cell())(word_embedding)
        rnn_layer_2 = keras.layers.Bidirectional(self._rnn_cell())(rnn_layer_1)
        rnn_layer_3 = keras.layers.Bidirectional(self._rnn_cell())(rnn_layer_2)
        rnn_layer_3 = keras.layers.Dropout(self.dropout)(rnn_layer_3)
        pos_out = keras.layers.TimeDistributed(keras.layers.Dense(self.num_pos_labels,
                                                                  activation='softmax',
                                                                  name='POS output'))(rnn_layer_1)
        chunks_out = keras.layers.TimeDistributed(keras.layers.Dense(self.num_chunk_labels,
                                                                     activation='softmax',
                                                                     name='Chunk output')
                                                  )(rnn_layer_3)
        model = keras.Model(word_input, [pos_out, chunks_out])
        if optimizer is None:
            self.optimizer = tf.train.GradientDescentOptimizer(0.1)
        else:
            self.optimizer = optimizer
        model.compile(optimizer=self.optimizer,
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        self.model = model

    def load_embedding_weights(self, weights):
        """
        Load word embedding weights into the model embedding layer

        Args:
            weights (numpy.ndarray): 2D matrix of word weights
        """
        assert self.model is not None, 'Cannot assign weights, apply build() before trying to ' \
                                       'loading embedding weights '
        self.model.get_layer(name='embedding').set_weights([weights])

    def _rnn_cell(self):
        if self.use_gpu:
            rnn_cell = keras.layers.CuDNNLSTM(self.feature_size, return_sequences=True)
        else:
            rnn_cell = keras.layers.LSTM(self.feature_size, return_sequences=True)
        return rnn_cell

    def _embedding_layer(self):
        return keras.layers.Embedding(self.vocabulary_size, self.feature_size,
                                      name='embedding', mask_zero=self.use_gpu is False)

    def fit(self, x, y, batch_size=1, epochs=1, validation_data=None, callbacks=None):
        """
        Fit provided X and Y on built model

        Args:
            x: x samples
            y: y samples
            batch_size (int, optional): batch size per sample
            epochs (int, optional): number of epochs to run before ending training process
            validation_data (optional): x and y samples to validate at the end of the epoch
            callbacks (optional): additional callbacks to run with fitting
        """
        self.model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs,
                       validation_data=validation_data, callbacks=callbacks)

    def predict(self, x, batch_size=1):
        """
        Predict labels given x.

        Args:
            x: samples for inference
            batch_size (int, optional): forward pass batch size

        Returns:
            tuple of numpy arrays of pos and chunk labels
        """
        return self.model.predict(x=x, batch_size=batch_size)

    def chunk_inference_mode(self):
        """
        Convert model into chunking tagging inference mode.
        Model can only be used for inference for chunking after calling this method,
        re-build the model for other use.
        """
        self.model = keras.Model(self.model.input, self.model.output[-1])

    def pos_inference_mode(self):
        """
        Convert model into POS tagging inference mode.
        Model can only be used for inference for POS after calling this method,
        re-build the model for other use.
        """
        self.model = keras.Model(self.model.input, self.model.output[0])

    def save(self, filepath):
        """
        Save the model to disk

        Args:
            filepath (str): file name to save model
        """
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as fd:
            self.model.save_weights(fd.name)
            model_weights = fd.read()
        topology = {k: v for k, v in self.__dict__.items()}
        topology.pop('model')
        topology.pop('optimizer')
        data = {'model_weights': model_weights,
                'model_topology': topology}
        with open(filepath, 'wb') as fp:
            pickle.dump(data, fp)

    def load(self, filepath):
        """
        Load model from disk

        Args:
            filepath (str): file name of model
        """
        with open(filepath, 'rb') as fp:
            model_data = pickle.load(fp)
        topology = model_data['model_topology']
        self.build(topology['vocabulary_size'],
                   topology['num_pos_labels'],
                   topology['num_chunk_labels'],
                   topology['feature_size'],
                   topology['dropout'],
                   optimizer=None)
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as fd:
            fd.write(model_data['model_weights'])
            fd.flush()
            self.model.load_weights(fd.name)
