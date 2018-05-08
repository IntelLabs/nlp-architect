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
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils


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


class MultiTaskIntentModel(IntentExtractionModel):
    """
    Multi-Task Intent and Slot tagging model (with character embedding)
    """

    def __init__(self):
        super(MultiTaskIntentModel, self).__init__()

    def save(self, path):
        save_load_utils.save_all_weights(self.model, path)

    def load(self, path):
        save_load_utils.load_all_weights(self.model, path, include_optimizer=False)

    def build(self,
              sentence_length,
              word_length,
              num_labels,
              num_intent_labels,
              word_vocab_size,
              char_vocab_size,
              word_emb_dims=100,
              char_emb_dims=25,
              char_lstm_dims=25,
              tagger_lstm_dims=100,
              dropout=0.2,
              embedding_matrix=None):
        """
        Build a model

        Args:
            sentence_length (int): max sentence length
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
            embedding_matrix (dict, optional): external word embedding dictionary
        """
        if embedding_matrix is not None:
            # load pre-trained word embeddings into an Embedding layer
            # note that we set trainable = False so as to keep the embeddings fixed
            embedding_layer = Embedding(word_vocab_size,
                                        word_emb_dims,
                                        weights=[embedding_matrix],
                                        input_length=sentence_length,
                                        trainable=True,
                                        name='word_embedding_layer')
        else:
            # learn embeddings ourselves
            embedding_layer = Embedding(word_vocab_size, word_emb_dims,
                                        input_length=sentence_length,
                                        name='word_embedding_layer')

        # create word embedding input and embedding layer
        words_input = Input(shape=(sentence_length,), name='words_input')
        word_embeddings = embedding_layer(words_input)
        word_embeddings = Dropout(dropout)(word_embeddings)

        # create word character input and embeddings layer
        word_chars_input = Input(shape=(sentence_length, word_length), name='word_chars_input')
        char_embedding_layer = Embedding(char_vocab_size, char_emb_dims,
                                         input_length=word_length, name='char_embedding_layer')
        # apply embedding to each word
        char_embeddings = TimeDistributed(char_embedding_layer)(word_chars_input)
        # feed dense char vectors into BiLSTM
        char_embeddings = TimeDistributed(Bidirectional(LSTM(char_lstm_dims)))(char_embeddings)
        char_embeddings = Dropout(dropout)(char_embeddings)

        # first BiLSTM layer (used for intent classification)
        first_bilstm_layer = Bidirectional(
            LSTM(tagger_lstm_dims, return_sequences=True, return_state=True))
        first_lstm_out = first_bilstm_layer(word_embeddings)

        lstm_y_sequence = first_lstm_out[:1][0]  # save y states of the LSTM layer
        states = first_lstm_out[1:]
        hf, cf, hb, cb = states  # extract last hidden states
        h_state = concatenate([hf, hb], axis=-1)
        intent_out = Dense(num_intent_labels, activation='softmax',
                           name='intent_classifier_output')(h_state)

        # create the 2nd feature vectors
        combined_features = concatenate([lstm_y_sequence, char_embeddings], axis=-1)

        # 2nd BiLSTM layer for label classification
        second_bilstm_layer = Bidirectional(
                LSTM(tagger_lstm_dims, return_sequences=True))(combined_features)
        second_bilstm_layer = Dropout(dropout)(second_bilstm_layer)

        # feed BiLSTM vectors into CRF
        crf = CRF(num_labels, sparse_target=False)
        labels_out = crf(second_bilstm_layer)

        # compile the model
        model = Model(inputs=[words_input, word_chars_input],
                      outputs=[intent_out, labels_out])

        # define losses and metrics
        loss_f = {'intent_classifier_output': 'categorical_crossentropy',
                  'crf_1': crf.loss_function}
        metrics = {'intent_classifier_output': 'categorical_accuracy',
                   'crf_1': crf.accuracy}

        model.compile(loss=loss_f,
                      optimizer='adam',
                      metrics=metrics)
        self.model = model


class JointSequentialIntentModel(IntentExtractionModel):
    """
    Joint Intent classification and Slot tagging Model
    """

    def __init__(self):
        super(JointSequentialIntentModel, self).__init__()

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


class EncDecIntentModel(IntentExtractionModel):
    """
    Encoder Decoder Deep LSTM Tagger Model
    """

    def __init__(self):
        super(EncDecIntentModel, self).__init__()

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
