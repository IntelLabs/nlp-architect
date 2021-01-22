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
from collections.abc import Sequence

from torch import nn
import torch
from torch.nn import CrossEntropyLoss
torch.multiprocessing.set_sharing_strategy('file_system')
from transformers.models.deberta.modeling_deberta import (
    DebertaPreTrainedModel, 
    DebertaModel, 
    ContextPooler, 
    StableDropout,
    DebertaSelfOutput,
    DebertaIntermediate,
    DebertaOutput,
    DebertaLayerNorm,
    DebertaAttention,
    DebertaLayer,
    DisentangledSelfAttention,
    DebertaEmbeddings,
    DebertaEncoder,
    c2p_dynamic_expand,
    p2c_dynamic_expand,
    pos_dynamic_expand,
    XSoftmax
)
from transformers.modeling_outputs import BaseModelOutput
from pytorch_lightning import _logger as log

class CustomDebertaAttention(DebertaAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = CustomDisentangledSelfAttention(config)
        self.output = DebertaSelfOutput(config)
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        self_output = self.self(
            hidden_states,
            attention_mask,
            return_att,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if return_att:
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)

        if return_att:
            return (attention_output, att_matrix)
        else:
            return attention_output

class CustomDebertaLayer(DebertaLayer):
    def __init__(self, config):
        super(DebertaLayer, self).__init__()
        self.attention = CustomDebertaAttention(config)
        self.intermediate = DebertaIntermediate(config)
        self.output = DebertaOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            return_att=return_att,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if return_att:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if return_att:
            return (layer_output, att_matrix)
        else:
            return layer_output

class CustomDisentangledSelfAttention(DisentangledSelfAttention):
    """
    Disentangled self-attention module

    Parameters:
        config (:obj:`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            `BertConfig`, for more details, please refer :class:`~transformers.DebertaConfig`

    """

    def __init__(self, config):
        super().__init__(config)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.in_proj = torch.nn.Linear(config.hidden_size, self.all_head_size * 3, bias=False)
        self.q_bias = torch.nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        self.v_bias = torch.nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []

        self.relative_attention = getattr(config, "relative_attention", False)
        self.talking_head = getattr(config, "talking_head", False)

        if self.talking_head:
            self.head_logits_proj = torch.nn.Linear(config.num_attention_heads, config.num_attention_heads, bias=False)
            self.head_weights_proj = torch.nn.Linear(
                config.num_attention_heads, config.num_attention_heads, bias=False
            )

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
                self.pos_proj = torch.nn.Linear(config.hidden_size, self.all_head_size, bias=False)
            if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
                self.pos_q_proj = torch.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        """
        Call the module

        Args:
            hidden_states (:obj:`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                `Attention(Q,K,V)`

            attention_mask (:obj:`torch.ByteTensor`):
                An attention mask matrix of shape [`B`, `N`, `N`] where `B` is the batch size, `N` is the maximum
                sequence length in which element [i,j] = `1` means the `i` th token in the input can attend to the `j`
                th token.

            return_att (:obj:`bool`, optional):
                Whether return the attention matrix.

            query_states (:obj:`torch.FloatTensor`, optional):
                The `Q` state in `Attention(Q,K,V)`.

            relative_pos (:obj:`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [`B`, `N`, `N`] with
                values ranging in [`-max_relative_positions`, `max_relative_positions`].

            rel_embeddings (:obj:`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [:math:`2 \\times
                \\text{max_relative_positions}`, `hidden_size`].


        """
        if query_states is None:
            qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
            query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
        else:

            def linear(w, b, x):
                if b is not None:
                    return torch.matmul(x, w.t()) + b.t()
                else:
                    return torch.matmul(x, w.t())  # + b.t()

            ws = self.in_proj.weight.chunk(self.num_attention_heads * 3, dim=0)
            qkvw = [torch.cat([ws[i * 3 + k] for i in range(self.num_attention_heads)], dim=0) for k in range(3)]
            qkvb = [None] * 3

            q = linear(qkvw[0], qkvb[0], query_states)
            k, v = [linear(qkvw[i], qkvb[i], hidden_states) for i in range(1, 3)]
            query_layer, key_layer, value_layer = [self.transpose_for_scores(x) for x in [q, k, v]]

        query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
        value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1 + len(self.pos_att_type)
        scale = math.sqrt(query_layer.size(-1) * scale_factor)
        query_layer = query_layer / scale
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)

        if rel_att is not None:
            attention_scores = attention_scores + rel_att

        # bxhxlxd
        if self.talking_head:
            attention_scores = self.head_logits_proj(attention_scores.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        if self.talking_head:
            attention_probs = self.head_weights_proj(attention_probs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_att:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), query_layer.device)
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.max_relative_positions)
        relative_pos = relative_pos.long().to(query_layer.device)
        rel_embeddings = rel_embeddings[
            self.max_relative_positions - att_span : self.max_relative_positions + att_span, :
        ].unsqueeze(0)
        if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)

        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos))
            score += c2p_att

        # position->content
        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            pos_query_layer /= math.sqrt(pos_query_layer.size(-1) * scale_factor)
            if query_layer.size(-2) != key_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), query_layer.device)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if "p2c" in self.pos_att_type:
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(
                p2c_att, dim=-1, index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)
            ).transpose(-1, -2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_dynamic_expand(pos_index, p2c_att, key_layer))
            score += p2c_att

        return score

class CustomDebertaEmbeddings(DebertaEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__(config)
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.output_to_half = False
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings


class CustomDebertaEncoder(DebertaEncoder):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([CustomDebertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, "relative_attention", False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.rel_embeddings = nn.Embedding(self.max_relative_positions * 2, config.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_module(
                next_kv,
                attention_mask,
                output_attentions,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )
            if output_attentions:
                hidden_states, att_m = hidden_states

            if query_states is not None:
                query_states = hidden_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = hidden_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

class CustomDebertaModel(DebertaModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = CustomDebertaEmbeddings(config)
        self.encoder = CustomDebertaEncoder(config)
        self.z_steps = 0
        self.config = config
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        VAT=False,
        perturbation=None,
        mode=None,
    ):
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

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        ### VAT ###
        if VAT and mode == "train":
            embedding_output += 0.001 * perturbation
        ###########

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    return_att=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )

class DebertaForTokenClassification(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = torch.nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.init_weights()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()


    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        VAT=False,
        mode=None,  # enable/disable VAT for train vs test
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        perturbation = 0  # overwrite if we are using VAT 
        if VAT and mode == "train":
            # Creating random perturbation
            B_ASP_LABEL = 1
            I_ASP_LABEL = 2
            aspect_idx_mask = (labels == B_ASP_LABEL).to(dtype=torch.float32) + (labels == I_ASP_LABEL).to(dtype=torch.float32)
            aspect_idx_mask = torch.unsqueeze(aspect_idx_mask, 2)
            aspect_idx_mask = aspect_idx_mask.repeat(1, 1, self.config.hidden_size)
            r_adv = torch.rand(aspect_idx_mask.size(), dtype=torch.float32, requires_grad=True).to(aspect_idx_mask)
            r_adv_masked = r_adv * aspect_idx_mask  # masked to only perturb aspects
            r_adv_masked.retain_grad()

            # VAT - FIRST PASS (computing r_adv)
            outputs = self.deberta(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                VAT=VAT,
                perturbation=r_adv_masked,
                mode=mode,
            )

            sequence_output = outputs[0]

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

            # Compute new perturbation using gradient            
            loss.backward()
            with torch.no_grad():
                perturbation = aspect_idx_mask * (r_adv_masked + r_adv_masked.grad)
            self.zero_grad()

        # Normal pass, or for VAT - SECOND PASS (computing loss with x + r_adv)
        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            VAT=VAT,
            perturbation=perturbation,
            mode=mode,
        )

        sequence_output = outputs[0]

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

# IMPORTANT !!!
# pooler.dense.weight and classifier.weight are initialized slightly differently than in DebertaForTokenClassification (main model weights are the same)
# if you change self.deberta = CustomDebertaModel(config) -> self.deberta = DebertaModel(config) this causes the weights to be initialized exactly the same
class CustomDebertaForTokenClassification(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = CustomDebertaModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = torch.nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.init_weights()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()


    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

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

# Removal of other stuff affects the initialization of the pooler and classifier weight? Same issue as above.
class DifferentCustomDebertaForTokenClassification(DebertaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaModel(config)
        self.init_weights()

def log_tensor_stats(a, name):
    log.info(f"Var: {name}; Shape: {tuple(a.shape)}; [min, max]: [{a.min().item():.4f}, "\
        f"{a.max().item():.4f}]; mean: {a.mean().item():.4f}; median: {a.median().item():.4f}")
