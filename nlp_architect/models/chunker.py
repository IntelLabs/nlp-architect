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
from neon.initializers import GlorotUniform
from neon.layers import MergeMultistream

from neon.layers.layer import LookupTable, Reshape, Dropout, Affine
from neon.layers.recurrent import DeepBiLSTM
from neon.models import Model
from neon.transforms import Logistic, Tanh, Softmax

from nlp_architect.contrib.neon.layers import DataInput, TimeDistributedRecurrentLast, \
    TimeDistBiLSTM


class SequenceChunker(object):
    """
    Sequence chunker model (Neon based)

    Args:
        sentence_length (str): max sentence length
        token_vocab_size (int): word vocabulary size
        pos_vocab_size (int, optional): POS vocabulary size
        char_vocab_size (int, optional): characters vocabulary size
        max_char_word_length (int, optional): max word length in characters
        token_embedding_size (int, optional): word embedding dims
        pos_embedding_size (int, optional): POS embedding dims
        char_embedding_size (int, optional): character embedding dims
        num_labels (int, optional): number of output labels possible per token
        lstm_hidden_size (int, optional): LSTM hidden size
        num_lstm_layers (int, optional): number of LSTM layers
        use_external_embedding (bool, optional): input is provided as external word embedding
        dropout (float, optional): dropout rate
    """

    def __init__(self, sentence_length,
                 token_vocab_size,
                 pos_vocab_size=None,
                 char_vocab_size=None,
                 max_char_word_length=20,
                 token_embedding_size=None,
                 pos_embedding_size=None,
                 char_embedding_size=None,
                 num_labels=None,
                 lstm_hidden_size=100,
                 num_lstm_layers=1,
                 use_external_embedding=None,
                 dropout=0.5
                 ):

        init = GlorotUniform()
        tokens = []
        if use_external_embedding is None:
            tokens.append(LookupTable(vocab_size=token_vocab_size,
                                      embedding_dim=token_embedding_size,
                                      init=init,
                                      pad_idx=0))
        else:
            tokens.append(DataInput())
        tokens.append(Reshape((-1, sentence_length)))
        f_layers = [tokens]

        # add POS tag input
        if pos_vocab_size is not None and pos_embedding_size is not None:
            f_layers.append([
                LookupTable(vocab_size=pos_vocab_size,
                            embedding_dim=pos_embedding_size,
                            init=init,
                            pad_idx=0),
                Reshape((-1, sentence_length))
            ])

        # add Character RNN input
        if char_vocab_size is not None and char_embedding_size is not None:
            char_lut_layer = LookupTable(vocab_size=char_vocab_size,
                                         embedding_dim=char_embedding_size,
                                         init=init,
                                         pad_idx=0)
            char_nn = [char_lut_layer,
                       TimeDistBiLSTM(char_embedding_size, init, activation=Logistic(),
                                      gate_activation=Tanh(),
                                      reset_cells=True, reset_freq=max_char_word_length),
                       TimeDistributedRecurrentLast(timesteps=max_char_word_length),
                       Reshape((-1, sentence_length))]

            f_layers.append(char_nn)

        layers = []
        if len(f_layers) == 1:
            layers.append(f_layers[0][0])
        else:
            layers.append(MergeMultistream(layers=f_layers, merge="stack"))
            layers.append(Reshape((-1, sentence_length)))
        layers += [DeepBiLSTM(lstm_hidden_size, init, activation=Logistic(),
                              gate_activation=Tanh(),
                              reset_cells=True,
                              depth=num_lstm_layers),
                   Dropout(keep=dropout),
                   Affine(num_labels, init, bias=init, activation=Softmax())]
        self._model = Model(layers=layers)

    def fit(self, dataset, optimizer, cost, callbacks, epochs=10):
        """
        fit a model

        Args:
            dataset: train/test set of CONLL2000 dataset
            optimizer: optimizer (Neon based)
            cost: cost function (Neon based)
            callbacks: callbacks (Neon based)
            epochs (int, optional): number of epochs to train
        """
        self._model.fit(dataset,
                        optimizer=optimizer,
                        num_epochs=epochs,
                        cost=cost,
                        callbacks=callbacks)

    def predict(self, dataset):
        """
        predict output of given dataset

        Args:
            dataset: Neon based iterator

        Returns:
            prediction on given dataset
        """
        return self._model.get_outputs(dataset)

    def save(self, path):
        """
        Save model weights to path

        Args:
            path (str): path to weights file
        """
        self._model.save_params(path)

    def get_model(self):
        """
        Get model

        Returns:
            Neon model object
        """
        return self._model
