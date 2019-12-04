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
import logging
import os
import json
import collections
from typing import List, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from transformers import (BertForQuestionAnswering,
                          XLMForQuestionAnswering,
                          XLNetForQuestionAnswering,
                          ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
                          BertPreTrainedModel,
                          RobertaConfig,
                          RobertaModel)
from nlp_architect.models.transformers.quantized_bert import QuantizedBertForQuestionAnswering
from nlp_architect.data.utils_squad import whitespace_tokenize
from nlp_architect.models.transformers.base_model import TransformerBase
from nlp_architect.data.utils_squad import (SquadExample,
                                            RawResult,
                                            write_predictions,
                                            RawResultExtended,
                                            write_predictions_extended,
                                            _check_is_max_context,
                                            _improve_answer_span)
from nlp_architect.models.transformers.question_answering.utils_squad_evaluate\
    import EVAL_OPTS, main as evaluation_script


logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class RobertaForQuestionAnswering(BertPreTrainedModel):
    """RoBERTa question answering head.
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):

        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class TransformerQuestionAnswering(TransformerBase):
    """
    Transformer question answering model

    Args:
        model_type (str): transformer base model type
        max_answer_length (int): The maximum length of an answer that can be generated.
        n_best_size (int): The total number of n-best predictions to generate.
        version_2_with_negative (bool): If true, examples contain some that do not have an answer.
        null_score_diff_threshold (float): If null_score - best_non_null is greater
        than the threshold predict null max_query_length (int): The maximum number of
        tokens for the question.
    """
    MODEL_CLASS = {
        'roberta': RobertaForQuestionAnswering,
        'bert': BertForQuestionAnswering,
        'quant_bert': QuantizedBertForQuestionAnswering,
        'xlnet': XLNetForQuestionAnswering,
        'xlm': XLMForQuestionAnswering,
    }

    def __init__(
            self, model_type: str, max_answer_length: int, n_best_size: int,
            version_2_with_negative: bool, null_score_diff_threshold: float,
            max_query_length: int, *args, load_quantized: bool = False, **kwargs):
        assert model_type in self.MODEL_CLASS.keys(), "unsupported model type"
        self.num_labels = 2
        super(TransformerQuestionAnswering, self).__init__(
            model_type, num_labels=self.num_labels, *args, **kwargs)
        self.model_class = self.MODEL_CLASS[model_type]
        self.model_type = model_type
        if model_type == 'quant_bert' and load_quantized:
            self.model = self.model_class.from_pretrained(self.model_name_or_path, from_tf=bool(
                '.ckpt' in self.model_name_or_path), config=self.config, from_8bit=load_quantized)
        else:
            self.model = self.model_class.from_pretrained(self.model_name_or_path, from_tf=bool(
                '.ckpt' in self.model_name_or_path), config=self.config)
        self.n_best_size = n_best_size
        self.max_query_length = max_query_length
        self.max_answer_length = max_answer_length
        self.version_2_with_negative = version_2_with_negative
        self.null_score_diff_threshold = null_score_diff_threshold
        self.to(self.device, self.n_gpus)

    def convert_to_tensors(self,
                           examples: List[SquadExample],
                           evaluate: bool,
                           max_seq_length: int = 128,
                           doc_stride: int = 128) -> TensorDataset:
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=self.tokenizer,
                                                max_seq_length=max_seq_length,
                                                doc_stride=doc_stride,
                                                max_query_length=self.max_query_length,
                                                is_training=not evaluate)
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        if evaluate:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_example_index, all_cls_index, all_p_mask)
        else:
            all_start_positions = torch.tensor(
                [f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor(
                [f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_cls_index, all_p_mask)
        if evaluate:
            return dataset, features
        return dataset

    def train(self,
              train_data_set: DataLoader,
              dev_data_set: Union[DataLoader, List[DataLoader]] = None,
              dev_examples: List[SquadExample] = None,
              dev_features: List[InputFeatures] = None,
              gradient_accumulation_steps: int = 1,
              per_gpu_train_batch_size: int = 8,
              max_steps: int = -1,
              num_train_epochs: int = 3,
              max_grad_norm: float = 1.0,
              logging_steps: int = 50,
              save_steps: int = 100,
              data_dir: str = None):
        """
        Train a model

        Args:
            train_data_set (DataLoader): training data set
            dev_data_set (Union[DataLoader, List[DataLoader]], optional): development set.
            Defaults to None.
            test_data_set (Union[DataLoader, List[DataLoader]], optional): test set.
            Defaults to None.
            gradient_accumulation_steps (int, optional): num of gradient accumulation steps.
            Defaults to 1.
            per_gpu_train_batch_size (int, optional): per GPU train batch size. Defaults to 8.
            max_steps (int, optional): max steps. Defaults to -1.
            num_train_epochs (int, optional): number of train epochs. Defaults to 3.
            max_grad_norm (float, optional): max gradient normalization. Defaults to 1.0.
            logging_steps (int, optional): number of steps between logging. Defaults to 50.
            save_steps (int, optional): number of steps between model save. Defaults to 100.
        """
        t_total, num_train_epochs = self.get_train_steps_epochs(max_steps,
                                                                num_train_epochs,
                                                                gradient_accumulation_steps,
                                                                len(train_data_set))
        if self.optimizer is None and self.scheduler is None:
            logger.info("Loading default optimizer and scheduler")
            self.setup_default_optimizer(total_steps=t_total)
        train_batch_size = per_gpu_train_batch_size * max(1, self.n_gpus)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data_set))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    train_batch_size * gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(num_train_epochs, desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_data_set, desc="Train iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'start_positions': batch[3],
                    'end_positions': batch[4],
                    'token_type_ids': None if self.model_type == 'xlm' else batch[2]
                }
                if self.model_type in ['xlnet', 'xlm']:
                    inputs.update(
                        {'cls_index': batch[5], 'p_mask': batch[6]}
                    )
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.n_gpus > 1:
                    loss = loss.mean()
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        ds = [dev_data_set]
                        for d in ds:
                            eval_results = self._evaluate(d, dev_features)
                            pred_file, odds_file = self.compute_predictions(
                                examples=dev_examples, features=dev_features,
                                eval_results=eval_results, prefix=global_step)
                            results = self.evaluate_predictions(data_dir, pred_file, odds_file)
                            for key, value in results.items():
                                logger.info('eval_%s: %s.', str(key), str(value))
                        logger.info('lr = %s', str(self.scheduler.get_lr()[0]))
                        logger.info('loss = %s', str((tr_loss - logging_loss) / logging_steps))
                        logging_loss = tr_loss

                    if save_steps > 0 and global_step % save_steps == 0:
                        self.save_model_checkpoint(output_path=self.output_path,
                                                   name='checkpoint-{}'.format(global_step))
                if 0 < max_steps < global_step:
                    epoch_iterator.close()
                    break
            if 0 < max_steps < global_step:
                train_iterator.close()
                break

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    def _evaluate(self, dataset: DataLoader, features: InputFeatures):
        logger.info("***** Running inference *****")
        logger.info(" Batch size: {}".format(dataset.batch_size))
        logger.info("  Num examples = %d", len(dataset))
        eval_results = []
        for batch in tqdm(dataset, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': None if self.model_type == 'xlm' else batch[2]
                }
                example_indices = batch[3]
                if self.model_type in ['xlnet', 'xlm']:
                    inputs.update(
                        {'cls_index': batch[4], 'p_mask': batch[5]}
                    )
                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                if self.model_type in ['xlnet', 'xlm']:
                    # XLNet uses a more complex post-processing procedure
                    result = RawResultExtended(
                        unique_id=unique_id,
                        start_top_log_probs=(outputs[0][i]).detach().cpu().tolist(),
                        start_top_index=(outputs[1][i]).detach().cpu().tolist(),
                        end_top_log_probs=(outputs[2][i]).detach().cpu().tolist(),
                        end_top_index=(outputs[3][i]).detach().cpu().tolist(),
                        cls_logits=(outputs[4][i]).detach().cpu().tolist())
                else:
                    result = RawResult(
                        unique_id=unique_id,
                        start_logits=(outputs[0][i]).detach().cpu().tolist(),
                        end_logits=(outputs[1][i]).detach().cpu().tolist())
                eval_results.append(result)
        return eval_results

    def compute_predictions(self, eval_results, prefix, examples, features):
        output_prediction_file = os.path.join(
            self.output_path, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(
            self.output_path, "nbest_predictions_{}.json".format(prefix))
        if self.version_2_with_negative:
            output_null_log_odds_file = os.path.join(
                self.output_path, "null_odds_{}.json".format(prefix))
        else:
            output_null_log_odds_file = None

        if self.model_type in ['xlnet', 'xlm']:
            # XLNet uses a more complex post-processing procedure
            write_predictions_extended(
                examples, features, eval_results, self.n_best_size,
                self.max_answer_length, output_prediction_file,
                output_nbest_file, output_null_log_odds_file,
                self.model.config.start_n_top, self.model.config.end_n_top,
                self.version_2_with_negative, self.tokenizer, False)
        else:
            write_predictions(
                examples, features, eval_results, self.n_best_size,
                self.max_answer_length, self.do_lower_case, output_prediction_file,
                output_nbest_file, output_null_log_odds_file, False,
                self.version_2_with_negative, self.null_score_diff_threshold)
        return output_prediction_file, output_null_log_odds_file

    def evaluate_predictions(self, data_dir, output_prediction_file, output_null_log_odds_file):
        file_name = 'dev-v2.0.json' if self.version_2_with_negative else 'dev-v1.1.json'
        dev_file = os.path.join(data_dir, file_name)
        evaluate_options = EVAL_OPTS(
            data_file=dev_file,
            pred_file=output_prediction_file,
            na_prob_file=output_null_log_odds_file)
        results = evaluation_script(evaluate_options)
        return results

    def inference(self, examples: List[SquadExample], max_seq_length: int, batch_size: int = 64):
        """
        Run inference on given examples

        Args:
            examples (List[SquadExample]): examples
            batch_size (int, optional): batch size. Defaults to 64.

        Returns:
            logits
        """
        data_set, features = self.convert_to_tensors(
            examples, max_seq_length=max_seq_length, evaluate=True)
        inf_sampler = SequentialSampler(data_set)
        inf_dataloader = DataLoader(data_set, sampler=inf_sampler, batch_size=batch_size)
        logits = self._evaluate(inf_dataloader, features)
        self.compute_predictions(
            examples=examples, features=features, eval_results=logits, prefix='inf')

    @classmethod
    def load_model(
            cls, model_path: str, model_type: str,
            max_answer_length,
            n_best_size,
            version_2_with_negative,
            null_score_diff_threshold,
            max_query_length,
            *args, **kwargs):
        """
        Create a TransformerQuestionAnswering model from given path

        Args:
            model_path (str): path to model
            model_type (str): model type

        Returns:
            TransformerQuestionAnswering: model
        """
        # Load a trained model and vocabulary from given path
        if not os.path.exists(model_path):
            raise FileNotFoundError
        return cls(
            model_type=model_type,
            max_answer_length=max_answer_length,
            model_name_or_path=model_path,
            n_best_size=n_best_size,
            version_2_with_negative=version_2_with_negative,
            null_score_diff_threshold=null_score_diff_threshold,
            max_query_length=max_query_length,
            *args, **kwargs)

    def save_model(self, output_dir: str, save_checkpoint: bool = False, args=None):
        """
        Save model/tokenizer/arguments to given output directory

        Args:
            output_dir (str): path to output directory
            save_checkpoint (bool, optional): save as checkpoint. Defaults to False.
            args ([type], optional): arguments object to save. Defaults to None.
        """
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info("Saving model checkpoint to %s", output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        if not save_checkpoint:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            if args is not None:
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))


def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 sequence_a_is_doc=False):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    # cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)

    features = []
    for (example_index, example) in enumerate(tqdm(examples)):

        # if example_index % 100 == 0:
        #     logger.info('Converting %s/%s pos %s neg %s', example_index,
        #     len(examples), cnt_pos, cnt_neg)

        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer
            #  (0 for token which can be in an answer) Original TF implem also
            #  keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # XLNet: P SEP Q SEP CLS
            # Others: CLS Q SEP P SEP
            if not sequence_a_is_doc:
                # Query
                tokens += query_tokens
                segment_ids += [sequence_a_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)

                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                if not sequence_a_is_doc:
                    segment_ids.append(sequence_b_segment_id)
                else:
                    segment_ids.append(sequence_a_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            if sequence_a_is_doc:
                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

                tokens += query_tokens
                segment_ids += [sequence_b_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (
                        tok_start_position >= doc_start and tok_end_position <= doc_end
                ):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    if sequence_a_is_doc:
                        doc_offset = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            if example_index < 1:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and span_is_impossible:
                    logger.info("impossible example")
                if is_training and not span_is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible))
            unique_id += 1

    return features
