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


class SequenceTagger(object):
    """
    A sequence tagging model for POS and Chunks written in Tensorflow (and Keras) based on the
    paper 'Deep multi-task learning with low level tasks supervised at lower layers'.
    The model has 3 Bi-LSTM layers and outputs POS and Chunk tags.

    Args:
        use_cudnn (bool, optional): use GPU based model (CUDNNA cells)
    """

    def __init__(self, use_cudnn=False):
        self.vocabulary_size = None
        self.num_pos_labels = None
        self.num_chunk_labels = None
        self.char_vocab_size = None
        self.feature_size = None
        self.dropout = None
        self.max_word_len = None
        self.classifier = None
        self.optimizer = None
        self.model = None
        self.use_cudnn = use_cudnn

    def build(self,
              vocabulary_size,
              num_pos_labels,
              num_chunk_labels,
              char_vocab_size=None,
              max_word_len=25,
              feature_size=100,
              dropout=0.5,
              classifier='softmax',
              optimizer=None):
        """
        Build a chunker/POS model

        Args:
            vocabulary_size (int): the size of the input vocabulary
            num_pos_labels (int): the size of of POS labels
            num_chunk_labels (int): the sie of chunk labels
            char_vocab_size (int, optional): character vocabulary size
            max_word_len (int, optional): max characters in a word
            feature_size (int, optional): feature size - determines the embedding/LSTM layer \
                hidden state size
            dropout (float, optional): dropout rate
            classifier (str, optional): classifier layer, 'softmax' for softmax or 'crf' for \
                conditional random fields classifier. default is 'softmax'.
            optimizer (tensorflow.python.training.optimizer.Optimizer, optional): optimizer, if \
                None will use default SGD (paper setup)
        """
        self.vocabulary_size = vocabulary_size
        self.char_vocab_size = char_vocab_size
        self.num_pos_labels = num_pos_labels
        self.num_chunk_labels = num_chunk_labels
        self.max_word_len = max_word_len
        self.feature_size = feature_size
        self.dropout = dropout
        self.classifier = classifier

        word_emb_layer = tf.keras.layers.Embedding(self.vocabulary_size, self.feature_size,
                                                   name='embedding', mask_zero=False)
        word_input = tf.keras.layers.Input(shape=(None,))
        word_embedding = word_emb_layer(word_input)
        input_src = word_input
        features = word_embedding

        # add char input if present
        if self.char_vocab_size is not None:
            char_input = tf.keras.layers.Input(shape=(None, self.max_word_len))
            char_emb_layer = tf.keras.layers.Embedding(self.char_vocab_size, 30,
                                                       name='char_embedding',
                                                       mask_zero=False)
            char_embedding = char_emb_layer(char_input)
            char_embedding = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv1D(30, 3, padding='same'))(char_embedding)
            char_embedding = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalMaxPooling1D())(
                char_embedding)

            input_src = [input_src, char_input]
            features = tf.keras.layers.concatenate([word_embedding, char_embedding])

        rnn_layer_1 = tf.keras.layers.Bidirectional(self._rnn_cell(return_sequences=True))(
            features)
        rnn_layer_2 = tf.keras.layers.Bidirectional(self._rnn_cell(return_sequences=True))(
            rnn_layer_1)
        rnn_layer_3 = tf.keras.layers.Bidirectional(self._rnn_cell(return_sequences=True))(
            rnn_layer_2)

        # outputs
        pos_out = tf.keras.layers.Dense(self.num_pos_labels, activation='softmax',
                                        name='pos_output')(rnn_layer_1)
        losses = {'pos_output': 'categorical_crossentropy'}
        metrics = {'pos_output': 'categorical_accuracy'}

        if 'crf' in self.classifier:
            with tf.device('/cpu:0'):
                chunk_crf = CRF(self.num_chunk_labels, name='chunk_crf')
                rnn_layer_3_dense = tf.keras.layers.Dense(self.num_chunk_labels)(
                    tf.keras.layers.Dropout(self.dropout)(rnn_layer_3))
                chunks_out = chunk_crf(rnn_layer_3_dense)
                losses['chunk_crf'] = chunk_crf.loss
                metrics['chunk_crf'] = chunk_crf.viterbi_accuracy
        else:
            chunks_out = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.num_chunk_labels,
                                      activation='softmax'),
                name='chunk_out')(rnn_layer_3)
            losses['chunk_out'] = 'categorical_crossentropy'
            metrics['chunk_out'] = 'categorical_accuracy'

        model = tf.keras.Model(input_src, [pos_out, chunks_out])
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(0.001, clipnorm=5.)
        else:
            self.optimizer = optimizer
        model.compile(optimizer=self.optimizer,
                      loss=losses,
                      metrics=metrics)
        self.model = model

    def load_embedding_weights(self, weights):
        """
        Load word embedding weights into the model embedding layer

        Args:
            weights (numpy.ndarray): 2D matrix of word weights
        """
        assert self.model is not None, 'Cannot assign weights, apply build() before trying to ' \
                                       'loading embedding weights '
        emb_layer = self.model.get_layer(name='embedding')
        assert emb_layer.output_dim == weights.shape[1], 'embedding vectors shape mismatch'
        emb_layer.set_weights([weights])

    def _rnn_cell(self, **kwargs):
        if self.use_cudnn:
            rnn_cell = tf.keras.layers.CuDNNLSTM(self.feature_size, **kwargs)
        else:
            rnn_cell = tf.keras.layers.LSTM(self.feature_size, **kwargs)
        return rnn_cell

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

    def save(self, filepath):
        """
        Save the model to disk

        Args:
            filepath (str): file name to save model
        """
        topology = {k: v for k, v in self.__dict__.items()}
        topology.pop('model')
        topology.pop('optimizer')
        topology.pop('use_cudnn')
        save_model(self.model, topology, filepath)

    def load(self, filepath):
        """
        Load model from disk

        Args:
            filepath (str): file name of model
        """
        load_model(filepath, self)


class SequenceChunker(SequenceTagger):
    """
    A sequence Chunker model written in Tensorflow (and Keras) based SequenceTagger model.
    The model uses only the chunking output of the model.
    """

    def predict(self, x, batch_size=1):
        """
        Predict labels given x.

        Args:
            x: samples for inference
            batch_size (int, optional): forward pass batch size

        Returns:
            tuple of numpy arrays of chunk labels
        """
        model = tf.keras.Model(self.model.input, self.model.output[-1])
        return model.predict(x=x, batch_size=batch_size)


class SequencePOSTagger(SequenceTagger):
    """
        A sequence POS tagger model written in Tensorflow (and Keras) based SequenceTagger model.
        The model uses only the chunking output of the model.
        """

    def predict(self, x, batch_size=1):
        """
        Predict labels given x.

        Args:
            x: samples for inference
            batch_size (int, optional): forward pass batch size

        Returns:
            tuple of numpy arrays of POS labels
        """
        model = tf.keras.Model(self.model.input, self.model.output[0])
        return model.predict(x=x, batch_size=batch_size)
