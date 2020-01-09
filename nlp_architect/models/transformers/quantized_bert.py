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
# pylint: disable=bad-super-call
"""
Quantized BERT layers and model
"""

import logging
import os
import sys

import torch
from torch import nn
from transformers.modeling_bert import (
    ACT2FN,
    BertAttention,
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertIntermediate,
    BertLayer,
    BertLayerNorm,
    BertModel,
    BertOutput,
    BertPooler,
    BertPreTrainedModel,
    BertSelfAttention,
    BertSelfOutput,
)

from nlp_architect.nn.torch.quantization import (
    QuantizationConfig,
    QuantizedEmbedding,
    QuantizedLayer,
    QuantizedLinear,
)

logger = logging.getLogger(__name__)

QUANT_WEIGHTS_NAME = "quant_pytorch_model.bin"

QUANT_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bert-base-uncased": "https://nlp-architect-data.s3-us-west-2.amazonaws.com/models/transformers/bert-base-uncased.json",  # noqa: E501
    "bert-large-uncased": "https://nlp-architect-data.s3-us-west-2.amazonaws.com/models/transformers/bert-large-uncased.json",  # noqa: E501
}


def quantized_linear_setup(config, name, *args, **kwargs):
    """
    Get QuantizedLinear layer according to config params
    """
    try:
        quant_config = QuantizationConfig.from_dict(getattr(config, name))
        linear = QuantizedLinear.from_config(*args, **kwargs, config=quant_config)
    except AttributeError:
        linear = nn.Linear(*args, **kwargs)
    return linear


def quantized_embedding_setup(config, name, *args, **kwargs):
    """
    Get QuantizedEmbedding layer according to config params
    """
    try:
        quant_config = QuantizationConfig.from_dict(getattr(config, name))
        embedding = QuantizedEmbedding.from_config(*args, **kwargs, config=quant_config)
    except AttributeError:
        embedding = nn.Embedding(*args, **kwargs)
    return embedding


class QuantizedBertConfig(BertConfig):
    pretrained_config_archive_map = QUANT_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP


class QuantizedBertEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = quantized_embedding_setup(
            config, "word_embeddings", config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = quantized_embedding_setup(
            config, "position_embeddings", config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = quantized_embedding_setup(
            config, "token_type_embeddings", config.type_vocab_size, config.hidden_size
        )

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


class QuantizedBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = quantized_linear_setup(
            config, "attention_query", config.hidden_size, self.all_head_size
        )
        self.key = quantized_linear_setup(
            config, "attention_key", config.hidden_size, self.all_head_size
        )
        self.value = quantized_linear_setup(
            config, "attention_value", config.hidden_size, self.all_head_size
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


class QuantizedBertSelfOutput(BertSelfOutput):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = quantized_linear_setup(
            config, "attention_output", config.hidden_size, config.hidden_size
        )
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


class QuantizedBertAttention(BertAttention):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = QuantizedBertSelfAttention(config)
        self.output = QuantizedBertSelfOutput(config)

    def prune_heads(self, heads):
        raise NotImplementedError("pruning heads is not implemented for Quantized BERT")


class QuantizedBertIntermediate(BertIntermediate):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = quantized_linear_setup(
            config, "ffn_intermediate", config.hidden_size, config.intermediate_size
        )
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, str)
        ):  # noqa: F821
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act


class QuantizedBertOutput(BertOutput):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = quantized_linear_setup(
            config, "ffn_output", config.intermediate_size, config.hidden_size
        )
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


class QuantizedBertLayer(BertLayer):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = QuantizedBertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            logger.warning("Using QuantizedBertLayer as decoder was not tested.")
            self.crossattention = QuantizedBertAttention(config)
        self.intermediate = QuantizedBertIntermediate(config)
        self.output = QuantizedBertOutput(config)


class QuantizedBertEncoder(BertEncoder):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [QuantizedBertLayer(config) for _ in range(config.num_hidden_layers)]
        )


class QuantizedBertPooler(BertPooler):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = quantized_linear_setup(
            config, "pooler", config.hidden_size, config.hidden_size
        )
        self.activation = nn.Tanh()


class QuantizedBertPreTrainedModel(BertPreTrainedModel):
    config_class = QuantizedBertConfig
    base_model_prefix = "quant_bert"

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, QuantizedLinear, QuantizedEmbedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, from_8bit=False, **kwargs):
        """load trained model from 8bit model"""
        if not from_8bit:
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        config = kwargs.pop("config", None)
        output_loading_info = kwargs.pop("output_loading_info", False)

        # Load config
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs
            )

        # Load model
        model_file = os.path.join(pretrained_model_name_or_path, QUANT_WEIGHTS_NAME)

        # Instantiate model.
        model = cls(config)
        # Set model to initialize variables to be loaded from quantized
        # checkpoint which are None by Default
        model.eval()
        # Get state dict of model
        state_dict = torch.load(model_file, map_location="cpu")
        logger.info("loading weights file {}".format(model_file))

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if not hasattr(model, cls.base_model_prefix) and any(
            s.startswith(cls.base_model_prefix) for s in state_dict.keys()
        ):
            start_prefix = cls.base_model_prefix + "."
        if hasattr(model, cls.base_model_prefix) and not any(
            s.startswith(cls.base_model_prefix) for s in state_dict.keys()
        ):
            model_to_load = getattr(model, cls.base_model_prefix)

        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )

        if hasattr(model, "tie_weights"):
            model.tie_weights()  # make sure word embedding weights are still tied

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        return model

    def save_pretrained(self, save_directory):
        """save trained model in 8bit"""
        super().save_pretrained(save_directory)
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self
        model_to_save.toggle_8bit(True)
        output_model_file = os.path.join(save_directory, QUANT_WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.toggle_8bit(False)

    def toggle_8bit(self, mode: bool):
        def _toggle_8bit(module):
            if isinstance(module, QuantizedLayer):
                module.mode_8bit = mode

        self.apply(_toggle_8bit)
        if mode:
            training = self.training
            self.eval()
            self.train(training)


class QuantizedBertModel(QuantizedBertPreTrainedModel, BertModel):
    def __init__(self, config):
        # we only want BertForQuestionAnswering init to run to avoid unnecessary
        # initializations
        super(BertModel, self).__init__(config)

        self.embeddings = QuantizedBertEmbeddings(config)
        self.encoder = QuantizedBertEncoder(config)
        self.pooler = QuantizedBertPooler(config)

        self.apply(self.init_weights)


class QuantizedBertForSequenceClassification(
    QuantizedBertPreTrainedModel, BertForSequenceClassification
):
    def __init__(self, config):
        # we only want BertForQuestionAnswering init to run to avoid unnecessary
        # initializations
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = QuantizedBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = quantized_linear_setup(
            config, "head", config.hidden_size, self.config.num_labels
        )

        self.apply(self.init_weights)


class QuantizedBertForQuestionAnswering(QuantizedBertPreTrainedModel, BertForQuestionAnswering):
    def __init__(self, config):
        # we only want BertForQuestionAnswering init to run to avoid unnecessary
        # initializations
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = QuantizedBertModel(config)
        self.qa_outputs = quantized_linear_setup(
            config, "head", config.hidden_size, config.num_labels
        )

        self.apply(self.init_weights)


class QuantizedBertForTokenClassification(QuantizedBertPreTrainedModel, BertForTokenClassification):
    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = QuantizedBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = quantized_linear_setup(
            config, "head", config.hidden_size, config.num_labels
        )

        self.apply(self.init_weights)
