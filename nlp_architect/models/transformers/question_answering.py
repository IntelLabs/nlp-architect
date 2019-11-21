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
from typing import List, Union

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from transformers import (BertForQuestionAnswering,
                          XLMForQuestionAnswering,
                          XLNetForQuestionAnswering)

from nlp_architect.models.transformers.quantized_bert import QuantizedBertForQuestionAnswering
from nlp_architect.utils.utils_squad import SquadExample
from nlp_architect.models.transformers.base_model import TransformerBase
from nlp_architect.utils.utils_squad import InputFeatures
from nlp_architect.utils.utils_squad import (convert_examples_to_features,
                                             RawResult,
                                             write_predictions,
                                             RawResultExtended,
                                             write_predictions_extended)
from nlp_architect.utils.utils_squad_evaluate import EVAL_OPTS, main as evaluation_script


logger = logging.getLogger(__name__)


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
                        start_top_log_probs=to_list(outputs[0][i]),
                        start_top_index=to_list(outputs[1][i]),
                        end_top_log_probs=to_list(outputs[2][i]),
                        end_top_index=to_list(outputs[3][i]),
                        cls_logits=to_list(outputs[4][i]))
                else:
                    result = RawResult(
                        unique_id=unique_id,
                        start_logits=to_list(outputs[0][i]),
                        end_logits=to_list(outputs[1][i]))
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


def to_list(tensor):
    return tensor.detach().cpu().tolist()
