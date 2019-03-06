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


class IntentExtractionModel(object):
    """
    Intent Extraction model base class (using tf.keras)
    """

    def __init__(self):
        self.model = None

    def fit(self, x, y, epochs=1, batch_size=1, callbacks=None, validation=None):
        """
        Train a model given input samples and target labels.

        Args:
            x: input samples
            y: input sample labels
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
            x: samples to run through the model
            batch_size (:obj:`int`, optional): batch size:

        Returns:
            numpy.ndarray: predicted values by the model
        """
        assert self.model, 'Model was not initialized'
        return self.model.predict(x, batch_size=batch_size)

    def save(self, path, exclude=None):
        """
        Save model to path

        Args:
            path (str): path to save model
            exclude (list, optional): a list of object fields to exclude when saving
        """
        assert self.model, 'Model was not initialized'
        topology = {k: v for k, v in self.__dict__.items()}
        topology.pop('model')
        if exclude and isinstance(exclude, list):
            for x in exclude:
                topology.pop(x)
        save_model(self.model, topology=topology, filepath=path)

    def load(self, path):
        """
        Load a trained model

        Args:
            path (str): path to model file
        """
        load_model(path, self)

    @property
    def input_shape(self):
        """:obj:`tuple`:Get input shape"""
        return self.model.layers[0].input_shape

    @staticmethod
    def _create_input_embed(sentence_len, is_extern_emb, token_emb_size, vocab_size):
        if is_extern_emb:
            in_layer = e_layer = tf.keras.layers.Input(shape=(sentence_len, token_emb_size,),
                                                       dtype='float32', name='tokens_input')
        else:
            in_layer = tf.keras.layers.Input(shape=(sentence_len,),
                                             dtype='int32', name='tokens_input')
            e_layer = tf.keras.layers.Embedding(vocab_size, token_emb_size,
                                                input_length=sentence_len,
                                                name='embedding_layer')(in_layer)
        return in_layer, e_layer

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


class MultiTaskIntentModel(IntentExtractionModel):
    """
    Multi-Task Intent and Slot tagging model (using tf.keras)

    Args:
        use_cudnn (bool, optional): use GPU based model (CUDNNA cells)
    """

    def __init__(self, use_cudnn=False):
        super().__init__()
        self.model = None
        self.word_length = None
        self.num_labels = None
        self.num_intent_labels = None
        self.word_vocab_size = None
        self.char_vocab_size = None
        self.word_emb_dims = None
        self.char_emb_dims = None
        self.char_lstm_dims = None
        self.tagger_lstm_dims = None
        self.dropout = None
        self.use_cudnn = use_cudnn

    def build(self,
              word_length,
              num_labels,
              num_intent_labels,
              word_vocab_size,
              char_vocab_size,
              word_emb_dims=100,
              char_emb_dims=30,
              char_lstm_dims=30,
              tagger_lstm_dims=100,
              dropout=0.2):
        """
        Build a model

        Args:
            word_length (int): max word length (in characters)
            num_labels (int): number of slot labels
            num_intent_labels (int): number of intent classes
            word_vocab_size (int): word vocabulary size
            char_vocab_size (int): character vocabulary size
            word_emb_dims (int, optional): word embedding dimensions
            char_emb_dims (int, optional): character embedding dimensions
            char_lstm_dims (int, optional): character feature LSTM hidden size
            tagger_lstm_dims (int, optional): tagger LSTM hidden size
            dropout (float, optional): dropout rate
        """
        self.word_length = word_length
        self.num_labels = num_labels
        self.num_intent_labels = num_intent_labels
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.word_emb_dims = word_emb_dims
        self.char_emb_dims = char_emb_dims
        self.char_lstm_dims = char_lstm_dims
        self.tagger_lstm_dims = tagger_lstm_dims
        self.dropout = dropout

        words_input = tf.keras.layers.Input(shape=(None,), name='words_input')
        embedding_layer = tf.keras.layers.Embedding(self.word_vocab_size,
                                                    self.word_emb_dims, name='word_embedding')
        word_embeddings = embedding_layer(words_input)
        word_embeddings = tf.keras.layers.Dropout(self.dropout)(word_embeddings)

        # create word character input and embeddings layer
        word_chars_input = tf.keras.layers.Input(shape=(None, self.word_length),
                                                 name='word_chars_input')
        char_embedding_layer = tf.keras.layers.Embedding(self.char_vocab_size, self.char_emb_dims,
                                                         input_length=self.word_length,
                                                         name='char_embedding')
        # apply embedding to each word
        char_embeddings = char_embedding_layer(word_chars_input)
        # feed dense char vectors into BiLSTM
        char_embeddings = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Bidirectional(self._rnn_cell(self.char_lstm_dims)))(char_embeddings)
        char_embeddings = tf.keras.layers.Dropout(self.dropout)(char_embeddings)

        # first BiLSTM layer (used for intent classification)
        first_bilstm_layer = tf.keras.layers.Bidirectional(
            self._rnn_cell(self.tagger_lstm_dims, return_sequences=True, return_state=True))
        first_lstm_out = first_bilstm_layer(word_embeddings)

        lstm_y_sequence = first_lstm_out[:1][0]  # save y states of the LSTM layer
        states = first_lstm_out[1:]
        hf, _, hb, _ = states  # extract last hidden states
        h_state = tf.keras.layers.concatenate([hf, hb], axis=-1)
        intents = tf.keras.layers.Dense(self.num_intent_labels, activation='softmax',
                                        name='intent_classifier_output')(h_state)

        # create the 2nd feature vectors
        combined_features = tf.keras.layers.concatenate([lstm_y_sequence, char_embeddings],
                                                        axis=-1)

        # 2nd BiLSTM layer for label classification
        second_bilstm_layer = tf.keras.layers.Bidirectional(self._rnn_cell(self.tagger_lstm_dims,
                                                                           return_sequences=True))(
            combined_features)
        second_bilstm_layer = tf.keras.layers.Dropout(self.dropout)(second_bilstm_layer)
        bilstm_out = tf.keras.layers.Dense(self.num_labels)(second_bilstm_layer)

        # feed BiLSTM vectors into CRF
        with tf.device('/cpu:0'):
            crf = CRF(self.num_labels, name='intent_slot_crf')
            labels = crf(bilstm_out)

        # compile the model
        model = tf.keras.Model(inputs=[words_input, word_chars_input],
                               outputs=[intents, labels])

        # define losses and metrics
        loss_f = {'intent_classifier_output': 'categorical_crossentropy',
                  'intent_slot_crf': crf.loss}
        metrics = {'intent_classifier_output': 'categorical_accuracy',
                   'intent_slot_crf': crf.viterbi_accuracy}

        model.compile(loss=loss_f,
                      optimizer=tf.train.AdamOptimizer(),
                      metrics=metrics)
        self.model = model

    def _rnn_cell(self, units, **kwargs):
        if self.use_cudnn:
            rnn_cell = tf.keras.layers.CuDNNLSTM(units, **kwargs)
        else:
            rnn_cell = tf.keras.layers.LSTM(units, **kwargs)
        return rnn_cell

    # pylint: disable=arguments-differ
    def save(self, path):
        """
        Save model to path

        Args:
            path (str): path to save model
        """
        super().save(path, ['use_cudnn'])


class Seq2SeqIntentModel(IntentExtractionModel):
    """
    Encoder Decoder Deep LSTM Tagger Model (using tf.keras)
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.vocab_size = None
        self.tag_labels = None
        self.token_emb_size = None
        self.encoder_depth = None
        self.decoder_depth = None
        self.lstm_hidden_size = None
        self.encoder_dropout = None
        self.decoder_dropout = None

    def build(self,
              vocab_size,
              tag_labels,
              token_emb_size=100,
              encoder_depth=1,
              decoder_depth=1,
              lstm_hidden_size=100,
              encoder_dropout=0.5,
              decoder_dropout=0.5):
        """
        Build the model

        Args:
            vocab_size (int): vocabulary size
            tag_labels (int): number of tag labels
            token_emb_size (int, optional): token embedding vector size
            encoder_depth (int, optional): number of encoder LSTM layers
            decoder_depth (int, optional): number of decoder LSTM layers
            lstm_hidden_size (int, optional): LSTM layers hidden size
            encoder_dropout (float, optional): encoder dropout
            decoder_dropout (float, optional): decoder dropout
        """
        self.vocab_size = vocab_size
        self.tag_labels = tag_labels
        self.token_emb_size = token_emb_size
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.lstm_hidden_size = lstm_hidden_size
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout

        words_input = tf.keras.layers.Input(shape=(None,), name='words_input')
        emb_layer = tf.keras.layers.Embedding(self.vocab_size, self.token_emb_size,
                                              name='word_embedding')
        benc_in = emb_layer(words_input)

        assert self.encoder_depth > 0, 'Encoder depth must be > 0'
        for i in range(self.encoder_depth):
            bencoder = tf.keras.layers.LSTM(self.lstm_hidden_size, return_sequences=True,
                                            return_state=True,
                                            go_backwards=True, dropout=self.encoder_dropout,
                                            name='encoder_blstm_{}'.format(i))(benc_in)
            benc_in = bencoder[0]
        b_states = bencoder[1:]
        benc_h, bene_c = b_states

        decoder_inputs = benc_in
        assert self.decoder_depth > 0, 'Decoder depth must be > 0'
        for i in range(self.decoder_depth):
            decoder = \
                tf.keras.layers.LSTM(
                    self.lstm_hidden_size,
                    return_sequences=True,
                    name='decoder_lstm_{}'.format(i))(decoder_inputs, initial_state=[benc_h,
                                                                                     bene_c])
            decoder_inputs = decoder
        decoder_outputs = tf.keras.layers.Dropout(self.decoder_dropout)(decoder)
        decoder_predictions = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.tag_labels, activation='softmax'),
            name='decoder_classifier')(decoder_outputs)

        self.model = tf.keras.Model(words_input, decoder_predictions)
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])
