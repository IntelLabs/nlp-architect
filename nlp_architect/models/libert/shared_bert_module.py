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


import torch
from torch import nn
torch.multiprocessing.set_sharing_strategy('file_system')
from transformers import BertForTokenClassification
from path_patterns import AspectMatcher, PatternPredictor

class InhouseBertForABSA(BertForTokenClassification):
    """ A BERT model for our token classification task - ABSA term extraction (BIO tagging). 
    This in-house model can share features and functionalities between the pipelines of both `model_type=bert` and `model_type=libert`,
    e.g., encorporating auxiliary loss from an auxiliary task.  """
    def __init__(self, config):
        super().__init__(config)
        # add classifiers for auxliary tasks
        self.pattern_classifiers = nn.ModuleList([PatternPredictor(config) for _ in self.config.auxiliary_layers])
        self.asp_matchers = nn.ModuleList([AspectMatcher(config) for _ in self.config.auxiliary_layers])

    def call_bert(self, inputs, **kwargs):
        return self.bert(inputs, **kwargs) 
       
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=True,
        output_hidden_states=True,
        opinion_mask=None,
        patterns=None,
        tgt_asp_indices=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.call_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # change function output to be a dict
        output_dict = {}
        if len(outputs) > 2: # add hidden states and attention if they are here
            output_dict["hidden_states"], output_dict["attentions"] = outputs[2:]
        output_dict["logits"] = logits 
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
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
            output_dict["loss"] = loss

            # **** Auxiliary tasks ****
            total_aux_loss = 0.0
            if "pattern" in self.config.auxiliary_tasks:
                assert patterns is not None, "Given labels but not patterns for auxiliary task"
                patt_aux_loss_cumm = 0.0
                for layer, pattern_classifier in zip(self.config.auxiliary_layers, self.pattern_classifiers):
                    hidden_states = output_dict["hidden_states"][layer]
                    pattern_predictor_outputs = pattern_classifier(hidden_states,
                                                                   patterns=patterns,
                                                                   opinion_mask=opinion_mask)
                    output_dict.update(**pattern_predictor_outputs)
                    patt_aux_loss_cumm += pattern_predictor_outputs["patt_aux_loss"] if "patt_aux_loss" in pattern_predictor_outputs else 0.0
                total_aux_loss += patt_aux_loss_cumm
            
            if "matched-AT" in self.config.auxiliary_tasks:
                asp_match_aux_loss_cumm = 0.0
                for layer, asp_matcher in zip(self.config.auxiliary_layers, self.asp_matchers):
                    hidden_states = output_dict["hidden_states"][layer]
                    asp_matcher_outputs = asp_matcher(hidden_states, 
                                                      opinion_mask=opinion_mask,
                                                      tgt_asp_indices=tgt_asp_indices) 
                    output_dict.update(**asp_matcher_outputs)       # in case of many layers, override all logits but last layer; while accumulating the loss for all layers
                    asp_match_aux_loss_cumm += asp_matcher_outputs["asp_match_aux_loss"] if "asp_match_aux_loss" in asp_matcher_outputs else 0.0
                total_aux_loss += asp_match_aux_loss_cumm
 
            # Report sum of all auxiliary losses:
            if self.config.auxiliary_tasks:
                output_dict.update(total_aux_loss=total_aux_loss) 
                       
        return output_dict  # Mandatory keys: scores ; Optional: loss, hidden_states, attentions, ...