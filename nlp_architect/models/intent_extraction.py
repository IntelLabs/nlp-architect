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

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

from keras import Input, Model
from keras.layers import (LSTM, Bidirectional, Dense, Dropout, Embedding,
                          RepeatVector, TimeDistributed, concatenate)
from keras.models import load_model


class IntentExtractionModel(object):
    """
    Intent Extraction model base class
    """
    def __init__(self):
        self.model = None

    def fit(self, x, y, epochs=1, batch_size=1, callbacks=None, validation=None):
        """
        Train a model given input samples and target labels.

        Args:
            x (numpy.ndarray): input samples
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
            x (numpy.ndarray):
            batch_size (:obj:`int`, optional): batch size:

        Returns:
            numpy.ndarray: predicted values by the model
        """
        assert self.model, 'Model was not initialized'
        return self.model.predict(x, batch_size=batch_size)

    def save(self, path):
        """
        Save model to path
        Args:
            path (str): path to save model
        """
        assert self.model, 'Model was not initialized'
        self.model.save(path)

    def load(self, path):
        """
        Load a trained model

        Args:
            path (str): path to model file
        """
        self.model = load_model(path)

    @property
    def input_shape(self):
        """:obj:`tuple`:Get input shape"""
        return self.model.layers[0].input_shape

    @staticmethod
    def _create_input_embed(sentence_len, is_extern_emb, token_emb_size, vocab_size):
        if is_extern_emb:
            in_layer = e_layer = Input(shape=(sentence_len, token_emb_size,),
                                       dtype='float32', name='tokens_input')
        else:
            in_layer = Input(shape=(sentence_len,),
                             dtype='int32', name='tokens_input')
            e_layer = Embedding(vocab_size, token_emb_size,
                                input_length=sentence_len,
                                name='embedding_layer')(in_layer)
        return in_layer, e_layer


class JointSequentialLSTM(IntentExtractionModel):
    """
    Joint Intent classification and Slot tagging Model
    """
    def __init__(self):
        super(JointSequentialLSTM, self).__init__()

    def build(self,
              sentence_length,
              vocab_size,
              tag_labels,
              intent_labels,
              token_emb_size=100,
              tagger_hidden=100,
              tagger_dropout=0.5,
              intent_classifier_hidden=100,
              emb_model_path=None):
        """
        Build the model

        Args:
            sentence_length (int): max length of a sentence
            vocab_size (int): vocabulary size
            tag_labels (int): number of tag labels
            intent_labels (int): number of intent labels
            token_emb_size (int): token embedding vectors size
            tagger_hidden (int): label tagger LSTM hidden size
            tagger_dropout (float): label tagger dropout rate
            intent_classifier_hidden (int): intent LSTM hidden size
            emb_model_path (str): external embedding model path
        """
        tokens_input, token_emb = self._create_input_embed(sentence_length,
                                                           emb_model_path is not None,
                                                           token_emb_size,
                                                           vocab_size)
        intent_enc = Bidirectional(LSTM(intent_classifier_hidden))(token_emb)
        intent_out = Dense(intent_labels, activation='softmax',
                           name='intent_classifier')(intent_enc)
        intent_vec_rep = RepeatVector(sentence_length)(intent_out)

        slot_emb = Bidirectional(LSTM(tagger_hidden, return_sequences=True))(token_emb)
        tagger_features = concatenate([slot_emb, intent_vec_rep], axis=-1)
        tagger = Bidirectional(
            LSTM(tagger_hidden, return_sequences=True))(tagger_features)
        tagger = Dropout(tagger_dropout)(tagger)
        tagger_out = TimeDistributed(
            Dense(tag_labels, activation='softmax'),
            name='slot_tag_classifier')(tagger)

        self.model = Model(inputs=tokens_input, outputs=[
            intent_out, tagger_out])
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                           loss_weights=[1., 1.], metrics=['categorical_accuracy'])


class EncDecTaggerModel(IntentExtractionModel):
    """
    Encoder Decoder Deep LSTM Tagger Model
    """
    def __init__(self):
        super(EncDecTaggerModel, self).__init__()

    def build(self,
              sentence_length,
              vocab_size,
              tag_labels,
              token_emb_size=100,
              encoder_depth=1,
              decoder_depth=1,
              lstm_hidden_size=100,
              encoder_dropout=0.5,
              decoder_dropout=0.5,
              emb_model_path=None):
        """
        Build the model

        Args:
            sentence_length (int): max sentence length
            vocab_size (int): vocabulary size
            tag_labels (int): number of tag labels
            token_emb_size (int, optional): token embedding vector size
            encoder_depth (int, optional): number of encoder LSTM layers
            decoder_depth (int, optional): number of decoder LSTM layers
            lstm_hidden_size (int, optional): LSTM layers hidden size
            encoder_dropout (float, optional): encoder dropout
            decoder_dropout (float, optional): decoder dropout
            emb_model_path (str, optional): external embedding model path
        """
        tokens_input, token_emb = self._create_input_embed(sentence_length,
                                                           emb_model_path is not None,
                                                           token_emb_size,
                                                           vocab_size)
        benc_in = token_emb
        assert encoder_depth > 0, 'Encoder depth must be > 0'
        for i in range(encoder_depth):
            bencoder = LSTM(lstm_hidden_size, return_sequences=True, return_state=True,
                            go_backwards=True, dropout=encoder_dropout,
                            name='encoder_blstm_{}'.format(i))(benc_in)
            benc_in = bencoder[0]
        b_states = bencoder[1:]
        benc_h, bene_c = b_states

        decoder_inputs = token_emb
        assert decoder_depth > 0, 'Decoder depth must be > 0'
        for i in range(decoder_depth):
            decoder = LSTM(lstm_hidden_size, return_sequences=True,
                           name='decoder_lstm_{}'.format(i))(decoder_inputs,
                                                             initial_state=[benc_h,
                                                                            bene_c])
            decoder_inputs = decoder
        decoder_outputs = Dropout(decoder_dropout)(decoder)
        decoder_predictions = TimeDistributed(
            Dense(tag_labels, activation='softmax'),
            name='decoder_classifier')(decoder_outputs)

        self.model = Model(tokens_input, decoder_predictions)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])
