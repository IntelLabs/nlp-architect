# ******************************************************************************
# Copyright 2019-2020 Intel Corporation
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

# pylint: disable=no-member, not-callable, arguments-differ, missing-class-docstring, too-many-locals, too-many-arguments, abstract-method
# pylint: disable=missing-module-docstring, missing-function-docstring, too-many-statements, too-many-instance-attributes

import math
from torch import nn
import torch
from torch.nn import CrossEntropyLoss
torch.multiprocessing.set_sharing_strategy('file_system')
from transformers.modeling_bert import (
    BertEncoder,
    BertLayer,
    BertAttention, 
    BertSelfAttention, 
    BertSelfOutput,
    BertConfig,
)
from transformers.modeling_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.activations import ACT2FN
from transformers import BertForTokenClassification, BertModel
from pytorch_lightning import _logger as log

class CustomBertConfig(BertConfig):
    def __init__(self, **kwargs):
        super().__init__()

    def add_extra_args(self, hparams):
        # pylint: disable=attribute-defined-outside-init
        self.custom_layers = hparams.custom_layers
        self.add_norm = hparams.add_norm

class CustomBertForTokenClassification(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = CustomBertModel(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        external_embeddings=None,
        regress_to_bert=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            external_embeddings=external_embeddings,
            regress_to_bert=regress_to_bert,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

        #if not return_dict:
        #    output = (logits,) + outputs[2:]
        #    return ((loss,) + output) if loss is not None else output

        #return TokenClassifierOutput(
        #    loss=loss,
        #    logits=logits,
        #    hidden_states=outputs.hidden_states,
        #    attentions=outputs.attentions,
        #)

class CustomBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        # Model weights are loaded in the call to super().__init__(config)
        # State dict still maps to the layers defined in the Custom Encoder
        self.encoder = CustomBertEncoder(config)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        external_embeddings=None,
        regress_to_bert=False,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            external_embeddings=external_embeddings,
            regress_to_bert=regress_to_bert,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        #return BaseModelOutputWithPooling(
        #    last_hidden_state=sequence_output,
        #    pooler_output=pooled_output,
        #    hidden_states=encoder_outputs.hidden_states,
        #    attentions=encoder_outputs.attentions,
        #)

class CustomBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([CustomBertLayer(config, base_bert=False) if layer_num in config.custom_layers else CustomBertLayer(config, base_bert=True) for layer_num in range(12)])
        #self.layer = nn.ModuleList([CustomBertLayer(config) for _ in range(12)])
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        external_embeddings=None,
        regress_to_bert=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    external_embeddings,
                    regress_to_bert,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    external_embeddings,
                    regress_to_bert,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        #return BaseModelOutput(
        #    last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        #)

        
class CustomBertLayer(BertLayer):
    def __init__(self, config, base_bert):
        super().__init__(config)
        self.base_bert = base_bert
        if not self.base_bert:
            self.external_attention = ExternalEmbeddingAttention(config)  # SEE BELOW but can be arbitrary module
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        external_embeddings=None,
        regress_to_bert=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]  # (batch_size, seq_len, 768)
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights
        
        # New
        if hasattr(self, "external_attention"):
            external_attention_output = self.external_attention(attention_output, external_embeddings=external_embeddings, regress_to_bert=regress_to_bert)[0]
            attention_output = external_attention_output
        
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        #external_attention_output = self.external_attention(attention_output, external_embeddings=external_embeddings)[0]
        #intermediate_output = self.intermediate(external_attention_output)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class ExternalEmbeddingSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        external_embeddings=None,
        regress_to_bert=False,
    ):
        if regress_to_bert == True:
            external_embeddings_copy = torch.Tensor(external_embeddings.size())
            external_embeddings_copy.copy_(external_embeddings).detach().cpu()
            assert torch.equal(external_embeddings_copy, torch.zeros(external_embeddings.size()).detach().cpu()), "Externally injected embeddings should be all 0"
            query_layer = hidden_states
            external_keys = external_embeddings
            external_values = external_embeddings
            token_keys = hidden_states
            token_values = hidden_states
        else:
            query_layer = self.query(hidden_states)  # (batch_size, 14, 768)
            external_keys = self.key(external_embeddings)      # external_embeddings = (batch_size, num_phrases, 768)
            external_values = self.value(external_embeddings)
            token_keys = self.key(hidden_states)
            token_values = self.value(hidden_states)
        
        batch_context_layer = []
        batch_attention_probs = []
        
        # Extracting each sequence in the batch
        # sequence_query, sequence_key, sequence_value: (seq_len, 768)
        # external_keys, external_values: (num_phases, 768)
        for sequence_query, sequence_key, sequence_value, external_key, external_value in zip(query_layer, token_keys, token_values, external_keys, external_values):
            sequence_context_layer = []
            sequence_attention_probs = []
            
            # Extracting query, key, and value for each token embedding in sequence
            for token_query, token_key, token_value in zip(sequence_query, sequence_key, sequence_value):
                
                # Creating new key and value vectors/matrices with the token embedding + external embeddings
                key_layer = torch.cat((token_key.unsqueeze(0), external_key), dim=0)        # (1 + num_phrases, 768)
                value_layer = torch.cat((token_value.unsqueeze(0), external_value), dim=0)  # (1 + num_phrases, 768)
                
                # Computing external Attention scores
                token_attention_scores = torch.matmul(token_query, key_layer.transpose(0,1))  # (1 + num_phrases)               
                token_attention_probs = nn.Softmax(dim=-1)(token_attention_scores)
                # Add dropout?: attention_probs = self.dropout(attention_probs)
                sequence_attention_probs.append(token_attention_probs)
                
                # New contextual embeddings with external information
                context_layer = torch.matmul(token_attention_probs, value_layer)  # (768)
                sequence_context_layer.append(context_layer)
            
            # Converting list to tensors
            sequence_attention_probs = torch.stack(sequence_attention_probs) # (seq_len, 1 + num_phrases)
            sequence_context_layer = torch.stack(sequence_context_layer)    # (seq_len, 768)
            batch_attention_probs.append(sequence_attention_probs)
            batch_context_layer.append(sequence_context_layer)

        # Converting list to tensors
        batch_attention_probs = torch.stack(batch_attention_probs)
        batch_context_layer = torch.stack(batch_context_layer)
           
        ################################
        
        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = (batch_context_layer, batch_attention_probs) if output_attentions else (batch_context_layer,)
        return outputs


class ExternalEmbeddingSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ExternalEmbeddingAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = ExternalEmbeddingSelfAttention(config)
        if config.add_norm:
            self.output = ExternalEmbeddingSelfOutput(config)
        self.pruned_heads = set()

        # MLP
        #self.mlp = MLP(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        external_embeddings=None,
        regress_to_bert=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            external_embeddings,
            regress_to_bert,
        )

        
        # Config choice to include Add + Norm block (dense, dropout, reisidual + layernorm)
        if hasattr(self, "output"):
            attention_output = self.output(self_outputs[0], hidden_states)
            outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        else:
            outputs = self_outputs


        # Adding MLP
        #outputs = self.mlp(outputs[0])

        return outputs

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        

    def forward(self, hidden_states):
        input_tensor = hidden_states
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


#def log_tensor_stats(a, name):
#    log.info(f"Var: {name}; Shape: {tuple(a.shape)}; [min, max]: [{a.min().item():.4f}, "\
#        f"{a.max().item():.4f}]; mean: {a.mean().item():.4f}; median: {a.median().item():.4f}")
