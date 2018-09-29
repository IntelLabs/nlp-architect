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

import numpy as np
from keras import Input, Model
from keras.layers import Embedding, Dropout, TimeDistributed, Bidirectional, LSTM, concatenate, \
    Dense
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils

from nlp_architect.utils.embedding import load_word_embeddings


class NERCRF(object):
    """
    NER model with CRF classification layer (Keras/TF implementation)
    """
    def __init__(self):
        self.model = None

    def build(self,
              sentence_length,
              target_label_dims,
              word_vocab,
              word_vocab_size,
              word_embedding_dims=100,
              tagger_lstm_dims=100,
              tagger_fc_dims=100,
              dropout=0.2,
              external_embedding_model=None):
        """
        Build a NERCRF model

        Args:
            sentence_length (int): max sentence length
            word_length (int): max word length in characters
            target_label_dims (int): number of entity labels (for classification)
            word_vocab (dict): word to int dictionary
            word_vocab_size (int): word vocabulary size
            word_embedding_dims (int): word embedding dimensions
            tagger_lstm_dims (int): word tagger LSTM output dimensions
            tagger_fc_dims (int): output fully-connected layer size
            dropout (float): dropout rate
            external_embedding_model (str): path to external word embedding model
        """
        # build word input
        words_input = Input(shape=(sentence_length,), name='words_input')

        if external_embedding_model is not None:
            # load and prepare external word embedding
            external_emb, ext_emb_size = load_word_embeddings(external_embedding_model)

            embedding_matrix = np.zeros((word_vocab_size, ext_emb_size))
            for word, i in word_vocab.items():
                embedding_vector = external_emb.get(word.lower())
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector

            # load pre-trained word embeddings into an Embedding layer
            # note that we set trainable = False so as to keep the embeddings fixed
            embedding_layer = Embedding(word_vocab_size,
                                        ext_emb_size,
                                        weights=[embedding_matrix],
                                        input_length=sentence_length,
                                        trainable=False)
        else:
            # learn embeddings ourselves
            embedding_layer = Embedding(word_vocab_size, word_embedding_dims,
                                        input_length=sentence_length)

        word_embeddings = embedding_layer(words_input)
        word_embeddings = Dropout(dropout)(word_embeddings)

        # encode using a bi-lstm
        bilstm = Bidirectional(LSTM(tagger_lstm_dims, return_sequences=True))(word_embeddings)
        bilstm = Dropout(dropout)(bilstm)
        after_lstm_hidden = Dense(tagger_fc_dims)(bilstm)

        # classify the dense vectors
        crf = CRF(target_label_dims, sparse_target=False)
        predictions = crf(after_lstm_hidden)

        # compile the model
        model = Model(inputs=words_input, outputs=predictions)
        model.compile(loss=crf.loss_function,
                      optimizer='adam',
                      metrics=[crf.accuracy])
        self.model = model

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
        save_load_utils.save_all_weights(self.model, path)

    def load(self, path):
        """
        Load model weights

        Args:
            path (str): path to load model from
        """
        save_load_utils.load_all_weights(self.model, path, include_optimizer=False)
