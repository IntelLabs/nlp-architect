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

# pylint: disable=no-member, not-callable, arguments-differ, missing-class-docstring, too-many-locals, too-many-arguments
# pylint: disable=missing-module-docstring, missing-function-docstring, too-many-statements, too-many-instance-attributes
import math
from torch import nn
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import BertEncoder, BertLayer, \
        BertAttention, BertSelfAttention, BertSelfOutput, BertConfig
from transformers import BertForTokenClassification, BertModel

class SaBertConfig(BertConfig):
    def __init__(self, **kwargs):
        super().__init__()

    def add_extra_args(self, hparams):
        # pylint: disable=attribute-defined-outside-init
        self.li_layer = hparams.li_layer
        self.replace_final = hparams.replace_final
        self.random_init = hparams.random_init
        self.all_layers = hparams.all_layers
        self.duplicated_rels = hparams.duplicated_rels
        self.transpose = hparams.transpose
        self.li_layers = hparams.li_layers

class SaBertForToken(BertForTokenClassification):
    def __init__(self, config):
        super(SaBertForToken, self).__init__(config)
        self.bert = SaBertModel(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, output_attentions=None,
                output_hidden_states=None, parse=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            parse=parse
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
                    active_loss, labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

class SaBertModel(BertModel):
    def __init__(self, config):
        super(SaBertModel, self).__init__(config)
        self.encoder = SaBertEncoder(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, output_attentions=None, output_hidden_states=None,
                parse=None):
        output_attentions = \
            output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None \
            else self.config.output_hidden_states

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

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length] ourselves in which
        # case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = \
            self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
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
        # input head_mask has shape [num_heads] or
        # [num_hidden_layers x num_heads] and head_mask is converted to shape
        # [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            parse=parse
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class SaBertEncoder(BertEncoder):
    def __init__(self, config):
        super(SaBertEncoder, self).__init__(config)
        self.layer = nn.ModuleList([SaBertLayer(config, layer_num) for \
            layer_num in range(config.num_hidden_layers)])
        self.li_layer = config.li_layer
        self.all_layers = config.all_layers
        self.li_layers = config.li_layers

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False, output_hidden_states=False, parse=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, bert_layer in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            parse_layer = None
            if self.all_layers or i == self.li_layer or i in self.li_layers:
                parse_layer = parse

            layer_outputs = bert_layer(
                hidden_states,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                parse_layer,
                output_attentions
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class SaBertLayer(BertLayer):
    def __init__(self, config, layer_num):
        super(SaBertLayer, self).__init__(config)
        self.attention = SaBertAttention(config, layer_num)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                parse=None, output_attentions=False):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
            parse=parse
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

class SaBertAttention(BertAttention):
    def __init__(self, config, layer_num):
        super(SaBertAttention, self).__init__(config)
        self.self = SaBertSelfAttention(config, layer_num)
        self.output = SaBertSelfOutput(config, layer_num)
        self.pruned_heads = set()

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False, parse=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states,
                                 encoder_attention_mask, output_attentions, parse)
        attention_output = self.output(self_outputs[0], hidden_states, parse)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class SaBertSelfAttention(BertSelfAttention):
    def __init__(self, config, layer_num):
        super(SaBertSelfAttention, self).__init__(config)
        self.orig_num_attention_heads = config.num_attention_heads
        self.replace_final = config.replace_final
        self.random_init = config.random_init
        self.duplicated_rels = config.duplicated_rels
        self.transpose = config.transpose

        #self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        if  (layer_num == config.li_layer or config.all_layers is True \
             or layer_num in config.li_layers):
            self.num_attention_heads = 13
            self.extra_query = nn.Linear(config.hidden_size, self.attention_head_size)
            self.extra_key = nn.Linear(config.hidden_size, self.attention_head_size)
            self.extra_value = nn.Linear(config.hidden_size, self.attention_head_size)
            nn.init.normal_(self.extra_key.weight.data, mean=0, std=0.02)
            nn.init.normal_(self.extra_key.bias.data, mean=0, std=0.02)
            nn.init.normal_(self.extra_query.weight.data, mean=0, std=0.02)
            nn.init.normal_(self.extra_query.bias.data, mean=0, std=0.02)
            nn.init.normal_(self.extra_value.weight.data, mean=0, std=0.02)
            nn.init.normal_(self.extra_value.bias.data, mean=0, std=0.02)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False, parse=None):

        if parse is not None:
            mixed_query_layer = \
                torch.cat((self.query(hidden_states), self.extra_query(hidden_states)), 2)
        else:
            mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            if parse is not None:
                self.all_head_size = self.num_attention_heads * self.attention_head_size
                mixed_key_layer = \
                    torch.cat((self.key(hidden_states), self.extra_key(hidden_states)), 2)
                mixed_value_layer = \
                    torch.cat((self.value(hidden_states), self.extra_value(hidden_states)), 2)
            else:
                self.all_head_size = self.num_attention_heads * self.attention_head_size
                mixed_key_layer = self.key(hidden_states)
                mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if parse is not None and not self.random_init:
            #  duplicated heads across all matrix (one vector duplicated across matrix)
            if self.duplicated_rels is True:
                parse = parse.sum(1, keepdim=True)
                # duplicate sum vector
                parse = parse.repeat(1, 64, 1)

            parse_norm = parse / parse.max(2, keepdim=True)[0]
            parse_norm[torch.isnan(parse_norm)] = 0

           # _, indices = parse_norm.max(2)
           # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           # ex_head_attention_probs = torch.zeros(parse_norm.shape).to(device)

           # for batch, tokens in enumerate(indices):
           #     mask_matrix = torch.zeros([parse_norm.shape[1], parse_norm.shape[2]])
           #     i=0
           #     for token in tokens:
           #         if token != 0:
           #             mask_matrix[i][token] = 1.
           #             i=i+1

            #    ex_head_attention_probs[batch] = mask_matrix

            #if self.duplicated_rels is True:
            #    parse_norm = ex_head_attention_probs

            original_12head_attn_scores = attention_scores[:, :self.orig_num_attention_heads]
            original_12head_attn_scores = \
                original_12head_attn_scores / math.sqrt(self.attention_head_size)
            original_12head_attn_scores = original_12head_attn_scores + attention_mask
            original_12head_attn_probs = nn.Softmax(dim=-1)(original_12head_attn_scores)
            extra_head_attn = attention_scores[:, self.orig_num_attention_heads, :, :]
            parse_norm = parse_norm*8+ attention_mask.squeeze(1)

            if not self.replace_final:
                if self.transpose:
                    parse_norm = parse_norm.transpose(-1, -2)

                extra_head_scaled_attn = ((extra_head_attn *8) * parse_norm).unsqueeze(1)
                extra_head_scaled_attn = extra_head_scaled_attn + attention_mask
                extra_head_scaled_attn_probs = nn.Softmax(dim=-1)(extra_head_scaled_attn)
                attention_probs = \
                    torch.cat((original_12head_attn_probs, extra_head_scaled_attn_probs), 1)

            # if self.replace_final is True:
            #     attention_probs = \
            # torch.cat((original_12head_attn_probs, ex_head_attention_probs.unsqueeze(1)),1)

        if parse is None or self.random_init:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # Apply the attention mask is
                # (precomputed for all layers in BertModel forward() function)
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class SaBertSelfOutput(BertSelfOutput):
    def __init__(self, config, layer_num):
        super(SaBertSelfOutput, self).__init__(config)
        if  (layer_num == config.li_layer or config.all_layers is True \
             or layer_num in config.li_layers):
            self.original_num_attention_heads = config.num_attention_heads
            self.attention_head_size = int(config.hidden_size / self.original_num_attention_heads)
            self.dense_extra_head = nn.Linear(self.attention_head_size, config.hidden_size)

    def forward(self, hidden_states, input_tensor, parse=None):
        if parse is not None:
            original_hidden_vec_size = self.original_num_attention_heads * self.attention_head_size
            hidden_states = self.dense(hidden_states[:, :, :original_hidden_vec_size]) + \
                self.dense_extra_head(hidden_states[:, :, original_hidden_vec_size:])
                # add relational embedddings:
                # + relational_embeddings (shape: config.hidden_size)
        else:
            hidden_states = self.dense(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
