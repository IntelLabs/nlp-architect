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
from torch.nn.parameter import Parameter
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
        self.grl = hparams.grl
        self.beta = hparams.beta
        self.gamma = hparams.gamma
        self.learn_gamma = hparams.learn_gamma
        self.gamma_val = hparams.gamma_val

class CustomBertForTokenClassification(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = CustomBertModel(config)
        self.learn_gamma = config.learn_gamma
        if not config.learn_gamma:
            self.gamma = config.gamma_val
        else:
            if config.gamma:
                self.gamma = Parameter(torch.tensor(0.0))  # Gets pushed through sigmoid, so effectively gamma = 0.5
                #nn.init.zeros_(self.gamma)
                #self.register_parameter("gamma", self.gamma)
                print(f"initializing gamma with value: {self.gamma}")
                #if self.gamma.requires_grad: self.gamma.register_hook(lambda x: print(f"gamma grad: {x}"))
                self.gamma.register_hook(lambda x: x * 100)

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
        review_embeddings=None,
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
            review_embeddings=review_embeddings,
        )
        normal_outputs = outputs[0]
        reconstruction_loss = outputs[1]
        sequence_output = normal_outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + normal_outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                asp_loss = loss_fct(active_logits, active_labels)
            else:
                asp_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # Creating joint loss aspect extraction/cluster prediction loss
            if hasattr(self, 'gamma'):
                if self.learn_gamma:
                    gamma_eff = torch.sigmoid(self.gamma)
                    total_loss = 2 * ((gamma_eff * asp_loss) + (1-gamma_eff) * reconstruction_loss)
                else:
                    total_loss = asp_loss + self.gamma * reconstruction_loss
                    gamma_eff = self.gamma
            else:
                total_loss = asp_loss + reconstruction_loss
                gamma_eff = 1

            outputs = (total_loss,) + outputs + (asp_loss, reconstruction_loss, gamma_eff)

        return outputs  # total_loss, scores, (hidden_states), (attentions), asp_loss, reconstruction_loss

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
        review_embeddings=None,
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
            review_embeddings=review_embeddings,
        )
        regular_encoder_outputs = encoder_outputs[0]
        reconstruction_loss = encoder_outputs[1]
        sequence_output = regular_encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return ((sequence_output, pooled_output) + regular_encoder_outputs[1:], reconstruction_loss)

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
        self.layer = nn.ModuleList([CustomBertLayer(config, base_bert=False) if (layer_num+1) in config.custom_layers else CustomBertLayer(config, base_bert=True) for layer_num in range(12)])
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
        review_embeddings=None,
    ):
        reconstruction_loss = 0

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
                    review_embeddings,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    review_embeddings,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # New
                reconstruction_loss += layer_outputs[2]
            else:
                reconstruction_loss += layer_outputs[1]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Normalizing reconstruction loss by number of layers used
        reconstruction_loss = reconstruction_loss / len(self.config.custom_layers)

        if not return_dict:
            return (tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None), reconstruction_loss)
        #return BaseModelOutput(
        #    last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        #)

        
class CustomBertLayer(BertLayer):
    def __init__(self, config, base_bert):
        super().__init__(config)
        self.base_bert = base_bert
        if not self.base_bert:
            self.reconstruction_module = ReconstructionModule(config)  # SEE BELOW but can be arbitrary module
            if config.grl:
                self.grl = GradientReversalLayer(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        review_embeddings=None,
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
        
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        # New
        if hasattr(self, "reconstruction_module"):
            if hasattr(self, "grl"):
                #if layer_output.requires_grad: layer_output.register_hook(lambda x: print(f"after grl grad: {x}"))
                reconstruction_input = self.grl(layer_output)
                #if layer_output.requires_grad: layer_output.register_hook(lambda x: print(f"before grl grad: {x}"))
            else:
                reconstruction_input = layer_output

            # Masking (zeroing) PAD tokens
            #if attention_mask is not None:
            #    active_mask = (attention_mask.view(-1) == 0)                      # "Extended" attention mask has 0's for active tokens
            #    active_mask = active_mask.view((layer_output.size()[:2] + (1,)))  # Reshaping to (batch_size, seq_len, 1) for broadcast
            #    cp_input = torch.mul(cp_input, active_mask)                       # Zeroing pad tokens with mask

            reconstruction_loss = self.reconstruction_module(reconstruction_input, review_embeddings=review_embeddings)
        else:
            reconstruction_loss = 0
        
        outputs = (layer_output,) + outputs + (reconstruction_loss,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class GRLFunction(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, hidden_states):
        #ctx.save_for_backward(beta)
        return hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        #beta = ctx.saved_tensors
        #print(f"grad_output shape: {grad_output.size()}")
        return grad_output.neg()

class GradientReversalLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        #self.beta = torch.tensor(config.beta, dtype=torch.long, requires_grad=False)

    def forward(self, hidden_states):
        #return GRLFunction.apply(hidden_states, self.beta)
        return GRLFunction.apply(hidden_states)

class ReconstructionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        #self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense1 = nn.Linear(config.hidden_size, 384)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        #self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense2 = nn.Linear(384, config.hidden_size)
        #self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.reconstruction_loss = nn.MSELoss()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        review_embeddings=None,
    ):
        review_embeddings = review_embeddings.squeeze()
    
        # Normalizing sum
        #num_nonzero = (hidden_states != 0).sum(dim=1)
        sentence_vectors = torch.sum(hidden_states, dim=1)  # (batch_size, max_seq_len) TODO: remove padded terms?
        #sentence_vectors = torch.div(sentence_vectors, num_nonzero)

        # MLP
        sentence_vectors = self.dense1(sentence_vectors)
        sentence_vectors = self.intermediate_act_fn(sentence_vectors)
        review_embeddings_pred = self.dense2(sentence_vectors)  # (batch_size)
        loss = self.reconstruction_loss(review_embeddings_pred, review_embeddings)
        return loss


#def log_tensor_stats(a, name):
#    log.info(f"Var: {name}; Shape: {tuple(a.shape)}; [min, max]: [{a.min().item():.4f}, "\
#        f"{a.max().item():.4f}]; mean: {a.mean().item():.4f}; median: {a.median().item():.4f}")
