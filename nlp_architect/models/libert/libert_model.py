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
from transformers.modeling_bert import BertEncoder, BertLayer, \
        BertAttention, BertSelfAttention, BertSelfOutput, BertConfig, BertLayerNorm
from transformers import BertForTokenClassification, BertModel
from pytorch_lightning import _logger as log

REL_EMBED_SIZE = 64
REL_EXPER = 1

class LiBertConfig(BertConfig):
    def __init__(self, **kwargs):
        super().__init__()

    def add_extra_args(self, hparams):
        # pylint: disable=attribute-defined-outside-init
        self.li_layer = hparams.li_layer
        self.replace_final = hparams.replace_final
        self.baseline = hparams.baseline
        self.all_layers = hparams.all_layers
        self.duplicated_rels = hparams.duplicated_rels
        self.transpose = hparams.transpose
        self.li_layers = hparams.li_layers
        self.use_syntactic_rels = hparams.use_syntactic_rels

class LiBertForToken(BertForTokenClassification):
    def __init__(self, config):
        super(LiBertForToken, self).__init__(config)

        self.rel_embed_layer = nn.Embedding(52, REL_EMBED_SIZE, padding_idx=0)
        self.bert = LiBertModel(config, self.rel_embed_layer)

        if REL_EXPER == 2:
            self.classifier = nn.Linear(config.hidden_size + REL_EMBED_SIZE, config.num_labels)
            self.baseline = config.baseline

        self.RelLayerNorm = BertLayerNorm(REL_EMBED_SIZE, eps=config.layer_norm_eps)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, output_attentions=None,
                output_hidden_states=None, syn_heads=None, syn_rels=None):

        syn_rels = self.rel_embed_layer(syn_rels)
        syn_rels = self.RelLayerNorm(syn_rels)
        # syn_rels = torch.div(syn_rels, 4)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            syn_heads=syn_heads,
            syn_rels=syn_rels
        )
        
        sequence_output = outputs[0]

        if REL_EXPER == 2:
            rel_embeds = self.rel_embed_layer(syn_rels)
            rel_embeds = rel_embeds if not self.baseline else torch.zeros_like(rel_embeds)
            sequence_with_relations = torch.cat((sequence_output, rel_embeds), 2)
            sequence_with_relations = self.dropout(sequence_with_relations)
            logits = self.classifier(sequence_with_relations)

        else:
            sequence_output = self.dropout(sequence_output) # add/concatenate syn_rels here
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

class LiBertModel(BertModel):
    def __init__(self, config, rel_embed_layer):
        super(LiBertModel, self).__init__(config)
            
        self.encoder = LiBertEncoder(config, rel_embed_layer)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, output_attentions=None, output_hidden_states=None,
                syn_heads=None, syn_rels=None):
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
            syn_heads=syn_heads,
            syn_rels=syn_rels
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class LiBertEncoder(BertEncoder):
    def __init__(self, config, rel_embed_layer):
        super(LiBertEncoder, self).__init__(config)
        
        self.layer = nn.ModuleList([LiBertLayer(config, layer_num, rel_embed_layer) for \
            layer_num in range(config.num_hidden_layers)])
        self.li_layer = config.li_layer
        self.all_layers = config.all_layers
        self.li_layers = config.li_layers

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False, output_hidden_states=False, syn_heads=None, syn_rels=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, bert_layer in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            parse_layer = None
            if self.all_layers or i == self.li_layer or i in self.li_layers:
                parse_layer = syn_heads

            layer_outputs = bert_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                syn_heads=parse_layer,
                syn_rels=syn_rels,
                output_attentions=output_attentions
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

class LiBertLayer(BertLayer):
    def __init__(self, config, layer_num, rel_embed_layer):
        super(LiBertLayer, self).__init__(config)
        self.attention = LiBertAttention(config, layer_num, rel_embed_layer)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                syn_heads=None, syn_rels=None, output_attentions=False):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
            syn_heads=syn_heads, syn_rels=syn_rels
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

class LiBertAttention(BertAttention):
    def __init__(self, config, layer_num, rel_embed_layer):
        super(LiBertAttention, self).__init__(config)
        self.self = LiBertSelfAttention(config, layer_num)
        self.output = LiBertSelfOutput(config, layer_num, rel_embed_layer)
        self.pruned_heads = set()

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False, syn_heads=None, syn_rels=None):
        self_outputs = self.self(hidden_states=hidden_states, attention_mask=attention_mask,
                                 head_mask=head_mask, encoder_hidden_states=encoder_hidden_states,
                                 encoder_attention_mask=encoder_attention_mask, 
                                 output_attentions=output_attentions, syn_heads=syn_heads)
        attention_output = self.output(self_outputs[0], hidden_states, syn_heads, syn_rels)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class LiBertSelfAttention(BertSelfAttention):
    def __init__(self, config, layer_num):
        super(LiBertSelfAttention, self).__init__(config)
        self.orig_num_attention_heads = config.num_attention_heads
        self.replace_final = config.replace_final
        self.duplicated_rels = config.duplicated_rels
        self.transpose = config.transpose

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
                output_attentions=False, syn_heads=None):

        # If this head is to be injected with linguistic info, initialise random Q, K, V
        if syn_heads is not None:
            self.all_head_size = self.num_attention_heads * self.attention_head_size            
            mixed_query_layer = torch.cat((self.query(hidden_states), self.extra_query(hidden_states)), 2)
            mixed_key_layer = torch.cat((self.key(hidden_states), self.extra_key(hidden_states)), 2)
            mixed_value_layer = torch.cat((self.value(hidden_states), self.extra_value(hidden_states)), 2)
  
        else:
            self.all_head_size = self.num_attention_heads * self.attention_head_size
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))        

        if syn_heads is not None:   

            #  duplicated heads across all matrix (one vector duplicated across matrix)
            if self.duplicated_rels is True:
                syn_heads = syn_heads.sum(1, keepdim=True)
                # duplicate sum vector
                syn_heads = syn_heads.repeat(1,64,1)
                 
            parse_norm = syn_heads / syn_heads.max(2, keepdim=True)[0]
            parse_norm[torch.isnan(parse_norm)] = 0

            original_12head_attn_scores = attention_scores[:, :self.orig_num_attention_heads]
            original_12head_attn_scores = original_12head_attn_scores / math.sqrt(self.attention_head_size)
            original_12head_attn_scores = original_12head_attn_scores + attention_mask
            original_12head_attn_probs = nn.Softmax(dim=-1)(original_12head_attn_scores)

            extra_head_attn = attention_scores[:,self.orig_num_attention_heads,:,:] 
            parse_norm = parse_norm * 8 + attention_mask.squeeze(1)
            
            # if self.replace_final is False:

            if self.transpose:                
                parse_norm = parse_norm.transpose(-1, -2)                   

            extra_head_scaled_attn = ((extra_head_attn * 8) * parse_norm).unsqueeze(1)       
            extra_head_scaled_attn = extra_head_scaled_attn + attention_mask
            extra_head_scaled_attn_probs = nn.Softmax(dim=-1)(extra_head_scaled_attn)
            attention_probs = torch.cat((original_12head_attn_probs, extra_head_scaled_attn_probs), 1)
           
            # if self.replace_final is True:
            #     attention_probs = torch.cat((original_12head_attn_probs, ex_head_attention_probs.unsqueeze(1)),1)

        else:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
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

class LiBertSelfOutput(BertSelfOutput):
    def __init__(self, config, layer_num, rel_embed_layer):
        super(LiBertSelfOutput, self).__init__(config)
        
        self.baseline = config.baseline

        if  (layer_num == config.li_layer or config.all_layers is True \
             or layer_num in config.li_layers):
            self.original_num_attention_heads = config.num_attention_heads
            self.attention_head_size = int(config.hidden_size / self.original_num_attention_heads)
            self.use_syntactic_rels = config.use_syntactic_rels
            self.syn_rel_layer = rel_embed_layer

            if REL_EXPER == 3:
                self.dense_13_head = nn.Linear(self.attention_head_size + REL_EMBED_SIZE, config.hidden_size) # [128, 768]
            else:
                self.dense_13_head = nn.Linear(self.attention_head_size, config.hidden_size) # [64, 768]

    def forward(self, hidden_states, input_tensor, syn_heads=None, syn_rels=None):
        
        if syn_heads is None: 
            #~~~~ Standard Attention Head ~~~~#
            assert False
            hidden_states = self.dense(hidden_states)

        else:
            assert syn_rels is not None

            #~~~~ Linguisticially-Informed Head ~~~~#

            ####### Add Syntactic Relations #######
            hidden_size_12_heads = self.original_num_attention_heads * self.attention_head_size
            output_12_heads = hidden_states[:, :, :hidden_size_12_heads]
            output_13_head = hidden_states[:, :, hidden_size_12_heads:]

            dense_output_12_heads = self.dense(output_12_heads)

            if REL_EXPER == 2:
                lingustic_info = self.dense_13_head(output_13_head)

            else:
                # rel_embeds = self.syn_rel_layer(syn_rels)

                ###### Add Syntactic Relations #######
                if self.baseline:

                    if REL_EXPER == 3:
                        rel_pad = torch.zeros_like(syn_rels)
                        lingustic_info = self.dense_13_head(torch.cat((output_13_head, rel_pad), 2))
                    
                    if REL_EXPER == 1:
                        lingustic_info = self.dense_13_head(output_13_head)                    

                else:
                    if REL_EXPER == 3:
                        concat = torch.cat((output_13_head, syn_rels))
                        lingustic_info = self.dense_13_head(concat, 2)


                    if REL_EXPER == 1:
                        lingustic_info = self.dense_13_head(output_13_head + syn_rels)

            hidden_states = dense_output_12_heads + lingustic_info

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def log_tensor_stats(a, name):
    log.info(f"Var: {name}; Shape: {tuple(a.shape)}; [min, max]: [{a.min().item():.4f}, "\
        f"{a.max().item():.4f}]; mean: {a.mean().item():.4f}; median: {a.median().item():.4f}")