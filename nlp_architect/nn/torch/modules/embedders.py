# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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
from typing import List

import torch
from torch import nn as nn

from nlp_architect.utils.io import load_json_file
from nlp_architect.utils.text import n_letters


class CNNLSTM(nn.Module):
    """CNN-LSTM embedder (based on Ma and Hovy. 2016)

    Args:
        word_vocab_size (int): word vocabulary size
        num_labels (int): number of labels (classifier)
        word_embedding_dims (int, optional): word embedding dims
        char_embedding_dims (int, optional): character embedding dims
        cnn_kernel_size (int, optional): character CNN kernel size
        cnn_num_filters (int, optional): character CNN number of filters
        lstm_hidden_size (int, optional): LSTM embedder hidden size
        lstm_layers (int, optional): num of LSTM layers
        bidir (bool, optional): apply bi-directional LSTM
        dropout (float, optional): dropout rate
        padding_idx (int, optinal): padding number for embedding layers

    """

    def __init__(
        self,
        word_vocab_size: int,
        num_labels: int,
        word_embedding_dims: int = 100,
        char_embedding_dims: int = 16,
        cnn_kernel_size: int = 3,
        cnn_num_filters: int = 128,
        lstm_hidden_size: int = 100,
        lstm_layers: int = 2,
        bidir: bool = True,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ):
        super(CNNLSTM, self).__init__()
        self.word_embeddings = nn.Embedding(
            word_vocab_size, word_embedding_dims, padding_idx=padding_idx
        )
        self.char_embeddings = nn.Embedding(
            n_letters + 1, char_embedding_dims, padding_idx=padding_idx
        )
        self.conv1 = nn.Conv1d(
            in_channels=char_embedding_dims,
            out_channels=cnn_num_filters,
            kernel_size=cnn_kernel_size,
            padding=1,
        )
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=word_embedding_dims + cnn_num_filters,
            hidden_size=lstm_hidden_size,
            bidirectional=bidir,
            batch_first=True,
            num_layers=lstm_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(
            in_features=lstm_hidden_size * 2 if bidir else lstm_hidden_size, out_features=num_labels
        )
        self.num_labels = num_labels
        self.padding_idx = padding_idx

    def load_embeddings(self, embeddings):
        """
        Load pre-defined word embeddings

        Args:
            embeddings (torch.tensor): word embedding tensor
        """
        self.word_embeddings = nn.Embedding.from_pretrained(
            embeddings, freeze=False, padding_idx=self.padding_idx
        )

    def forward(self, words, word_chars, **kwargs):
        """
        CNN-LSTM forward step

        Args:
            words (torch.tensor): words
            word_chars (torch.tensor): word character tensors

        Returns:
            torch.tensor: logits of model
        """
        word_embeds = self.word_embeddings(words)
        char_embeds = self.char_embeddings(word_chars)
        saved_char_size = char_embeds.size()[:2]
        char_embeds = char_embeds.permute(0, 1, 3, 2)
        input_size = char_embeds.size()
        squashed_shape = [-1] + list(input_size[2:])
        char_embeds_reshape = char_embeds.contiguous().view(
            *squashed_shape
        )  # (samples * timesteps, input_size)
        char_embeds = self.conv1(char_embeds_reshape)
        char_embeds = char_embeds.permute(0, 2, 1)
        char_embeds = self.relu(char_embeds)
        char_embeds, _ = torch.max(char_embeds, 1)  # global max pooling
        new_size = saved_char_size + char_embeds.size()[1:]
        char_features = char_embeds.contiguous().view(new_size)

        features = torch.cat((word_embeds, char_features), -1)
        features = self.dropout(features)

        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(features)
        lstm_out = self.dropout(lstm_out)
        logits = self.dense(lstm_out)

        return logits

    @classmethod
    def from_config(cls, word_vocab_size: int, num_labels: int, config: str):
        """
        Load a model from a configuration file
        A valid configuration file is a JSON file with fields as in class `__init__`

        Args:
            word_vocab_size (int): word vocabulary size
            num_labels (int): number of labels (classifier)
            config (str): path to configuration file

        Returns:
            CNNLSTM: CNNLSTM module pre-configured
        """
        if not os.path.exists(config):
            raise FileNotFoundError
        cfg = load_json_file(config)
        return cls(word_vocab_size=word_vocab_size, num_labels=num_labels, **cfg)


class IDCNN(nn.Module):
    """
    ID-CNN (iterated dilated) tagging model (based on Strubell et al 2017) with word character
    embedding (using CNN feature extractors)

    Args:
        word_vocab_size (int): word vocabulary size
        num_labels (int): number of labels (classifier)
        word_embedding_dims (int, optional): word embedding dims
        char_embedding_dims (int, optional): character embedding dims
        char_cnn_filters (int, optional): character CNN kernel size
        char_cnn_kernel_size (int, optional): character CNN number of filters
        cnn_kernel_size (int, optional): CNN embedder kernel size
        cnn_num_filters (int, optional): CNN embedder number of filters
        input_dropout (float, optional): input dropout rate
        word_dropout (float, optional): pre embedder dropout rate
        hidden_dropout (float, optional): pre classifier dropout rate
        blocks (int, optinal): number of blocks
        dilations (List, optinal): List of dilations per CNN layer
        padding_idx (int, optinal): padding number for embedding layers

    """

    def __init__(
        self,
        word_vocab_size: int,
        num_labels: int,
        word_embedding_dims: int = 100,
        char_embedding_dims: int = 16,
        char_cnn_filters: int = 128,
        char_cnn_kernel_size: int = 3,
        cnn_kernel_size: int = 3,
        cnn_num_filters: int = 128,
        input_dropout: float = 0.5,
        word_dropout: float = 0.5,
        hidden_dropout: float = 0.5,
        blocks: int = 1,
        dilations: List = None,
        padding_idx: int = 0,
    ):
        super(IDCNN, self).__init__()
        if dilations is None:
            dilations = [1, 2, 1]
        self.num_blocks = blocks
        self.dilation = dilations
        self.padding = dilations
        self.num_labels = num_labels
        self.padding_idx = padding_idx
        self.word_embeddings = nn.Embedding(
            word_vocab_size, word_embedding_dims, padding_idx=padding_idx
        )
        self.conv1 = nn.Conv1d(
            in_channels=word_embedding_dims,
            out_channels=cnn_num_filters,
            kernel_size=cnn_kernel_size,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.cnv_layers = []

        for i in range(len(self.dilation)):
            self.cnv_layers.append(
                nn.Conv1d(
                    in_channels=cnn_num_filters,
                    out_channels=cnn_num_filters,
                    kernel_size=3,
                    padding=self.padding[i],
                    dilation=self.dilation[i],
                )
            )
        self.cnv_layers = nn.ModuleList(self.cnv_layers)
        self.dense = nn.Linear(
            in_features=(cnn_num_filters * self.num_blocks) + char_cnn_filters,
            out_features=num_labels,
        )

        self.char_embeddings = nn.Embedding(
            n_letters + 1, char_embedding_dims, padding_idx=padding_idx
        )
        self.char_conv = nn.Conv1d(
            in_channels=char_embedding_dims,
            out_channels=char_cnn_filters,
            kernel_size=char_cnn_kernel_size,
            padding=1,
        )
        self.i_drop = nn.Dropout(input_dropout)
        self.w_drop = nn.Dropout(word_dropout)
        self.h_drop = nn.Dropout(hidden_dropout)

    def forward(self, words, word_chars, **kwargs):
        """
        IDCNN forward step

        Args:
            words (torch.tensor): words
            word_chars (torch.tensor): word character tensors

        Returns:
            torch.tensor: logits of model
        """
        word_embeds = self.word_embeddings(words)
        word_embeds = self.i_drop(word_embeds)
        word_embeds = word_embeds.permute(0, 2, 1)
        conv1 = self.conv1(word_embeds)
        conv1 = self.w_drop(conv1)
        conv_outputs = []
        for _ in range(self.num_blocks):
            for j in range(len(self.cnv_layers)):
                conv1 = self.cnv_layers[j](conv1)
                if j == len(self.cnv_layers) - 1:
                    conv_outputs.append(conv1.permute(0, 2, 1))

        word_features = torch.cat(conv_outputs, 2)

        char_embeds = self.char_embeddings(word_chars)
        saved_char_size = char_embeds.size()[:2]
        char_embeds = char_embeds.permute(0, 1, 3, 2)
        input_size = char_embeds.size()
        squashed_shape = [-1] + list(input_size[2:])
        char_embeds_reshape = char_embeds.contiguous().view(*squashed_shape)
        char_embeds = self.char_conv(char_embeds_reshape)
        char_embeds = char_embeds.permute(0, 2, 1)
        char_embeds = self.relu(char_embeds)
        char_embeds, _ = torch.max(char_embeds, 1)  # global max pooling
        new_size = saved_char_size + char_embeds.size()[1:]
        char_features = char_embeds.contiguous().view(new_size)

        features = torch.cat((word_features, char_features), -1)
        features = self.h_drop(features)
        logits = self.dense(features)
        return logits

    @classmethod
    def from_config(cls, word_vocab_size: int, num_labels: int, config: str):
        """
        Load a model from a configuration file
        A valid configuration file is a JSON file with fields as in class `__init__`

        Args:
            word_vocab_size (int): word vocabulary size
            num_labels (int): number of labels (classifier)
            config (str): path to configuration file

        Returns:
            IDCNN: IDCNNEmbedder module pre-configured
        """
        if not os.path.exists(config):
            raise FileNotFoundError
        cfg = load_json_file(config)
        return cls(word_vocab_size=word_vocab_size, num_labels=num_labels, **cfg)

    def load_embeddings(self, embeddings):
        """
        Load pre-defined word embeddings

        Args:
            embeddings (torch.tensor): word embedding tensor
        """
        self.word_embeddings = nn.Embedding.from_pretrained(
            embeddings, freeze=False, padding_idx=self.padding_idx
        )
