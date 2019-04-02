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

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from nlp_architect.contrib.tensorflow.python.keras.layers.crf import CRF
from nlp_architect.contrib.tensorflow.python.keras.utils.layer_utils import load_model, save_model


class NERCRF(object):
    """
    Bi-LSTM NER model with CRF classification layer (tf.keras model)

    Args:
        use_cudnn (bool, optional): use cudnn LSTM cells
    """

    def __init__(self, use_cudnn=False):
        self.model = None
        self.word_length = None
        self.target_label_dims = None
        self.word_vocab_size = None
        self.char_vocab_size = None
        self.word_embedding_dims = None
        self.char_embedding_dims = None
        self.tagger_lstm_dims = None
        self.dropout = None
        self.use_cudnn = use_cudnn

    def build(self,
              word_length,
              target_label_dims,
              word_vocab_size,
              char_vocab_size,
              word_embedding_dims=100,
              char_embedding_dims=16,
              tagger_lstm_dims=200,
              dropout=0.5):
        """
        Build a NERCRF model

        Args:
            word_length (int): max word length in characters
            target_label_dims (int): number of entity labels (for classification)
            word_vocab_size (int): word vocabulary size
            char_vocab_size (int): character vocabulary size
            word_embedding_dims (int): word embedding dimensions
            char_embedding_dims (int): character embedding dimensions
            tagger_lstm_dims (int): word tagger LSTM output dimensions
            dropout (float): dropout rate
        """
        self.word_length = word_length
        self.target_label_dims = target_label_dims
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.word_embedding_dims = word_embedding_dims
        self.char_embedding_dims = char_embedding_dims
        self.tagger_lstm_dims = tagger_lstm_dims
        self.dropout = dropout

        # build word input
        words_input = tf.keras.layers.Input(shape=(None,), name='words_input')
        embedding_layer = tf.keras.layers.Embedding(self.word_vocab_size, self.word_embedding_dims,
                                                    name='word_embedding')
        word_embeddings = embedding_layer(words_input)

        # create word character embeddings
        word_chars_input = tf.keras.layers.Input(shape=(None, self.word_length),
                                                 name='word_chars_input')
        char_embedding_layer = tf.keras.layers.Embedding(self.char_vocab_size,
                                                         self.char_embedding_dims,
                                                         name='char_embedding')(word_chars_input)
        char_embeddings = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'))(
                char_embedding_layer)
        char_embeddings = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalMaxPooling1D())(
            char_embeddings)

        # create the final feature vectors
        features = tf.keras.layers.concatenate([word_embeddings, char_embeddings], axis=-1)

        # encode using a bi-LSTM
        features = tf.keras.layers.Dropout(self.dropout)(features)
        bilstm = tf.keras.layers.Bidirectional(self._rnn_cell(self.tagger_lstm_dims,
                                                              return_sequences=True))(features)
        bilstm = tf.keras.layers.Bidirectional(self._rnn_cell(self.tagger_lstm_dims,
                                                              return_sequences=True))(bilstm)
        bilstm = tf.keras.layers.Dropout(self.dropout)(bilstm)
        bilstm = tf.keras.layers.Dense(self.target_label_dims)(bilstm)

        inputs = [words_input, word_chars_input]

        sequence_lengths = tf.keras.layers.Input(shape=(1,), dtype='int32', name='seq_lens')
        inputs.append(sequence_lengths)
        crf = CRF(self.target_label_dims, name='ner_crf')
        predictions = crf(inputs=bilstm, sequence_lengths=sequence_lengths)

        # compile the model
        model = tf.keras.Model(inputs=inputs,
                               outputs=predictions)
        model.compile(loss={'ner_crf': crf.loss},
                      optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=5.))

        self.model = model

    def _rnn_cell(self, units, **kwargs):
        if self.use_cudnn:
            rnn_cell = tf.keras.layers.CuDNNLSTM(units, **kwargs)
        else:
            rnn_cell = tf.keras.layers.LSTM(units, **kwargs)
        return rnn_cell

    def load_embedding_weights(self, weights):
        """
        Load word embedding weights into the model embedding layer

        Args:
            weights (numpy.ndarray): 2D matrix of word weights
        """
        assert self.model is not None, 'Cannot assign weights, apply build() before trying to ' \
                                       'loading embedding weights '
        emb_layer = self.model.get_layer(name='word_embedding')
        assert emb_layer.output_dim == weights.shape[1], 'embedding vectors shape mismatch'
        emb_layer.set_weights([weights])

    def fit(self, x, y, epochs=1, batch_size=1, callbacks=None, validation=None):
        """
        Train a model given input samples and target labels.

        Args:
            x (numpy.ndarray or :obj:`numpy.ndarray`): input samples
            y (numpy.ndarray): input sample labels
            epochs (:obj:`int`, optional): number of epochs to train
            batch_size (:obj:`int`, optional): batch size
            callbacks(:obj:`Callback`, optional): Keras compatible callbacks
            validation(:obj:`list` of :obj:`numpy.ndarray`, optional): optional validation data
                to be evaluated when training
        """
        assert self.model, 'Model was not initialized'
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, shuffle=True,
                       validation_data=validation,
                       callbacks=callbacks)

    def predict(self, x, batch_size=1):
        """
        Get the prediction of the model on given input

        Args:
            x (numpy.ndarray or :obj:`numpy.ndarray`): input samples
            batch_size (:obj:`int`, optional): batch size

        Returns:
            numpy.ndarray: predicted values by the model
        """
        assert self.model, 'Model was not initialized'
        return self.model.predict(x, batch_size=batch_size)

    def save(self, path):
        """
        Save model to path

        Args:
            path (str): path to save model weights
        """
        topology = {k: v for k, v in self.__dict__.items()}
        topology.pop('model')
        topology.pop('use_cudnn')
        save_model(self.model, topology, path)

    def load(self, path):
        """
        Load model weights

        Args:
            path (str): path to load model from
        """
        load_model(path, self)
