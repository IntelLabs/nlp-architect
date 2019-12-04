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
import argparse
import logging
import os

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from nlp_architect.data.question_answering import QuestionAnsweringProcessor
from nlp_architect.nn.torch import setup_backend, set_seed
from nlp_architect.procedures.procedure import Procedure
from nlp_architect.procedures.registry import (register_run_cmd,
                                               register_train_cmd)
from nlp_architect.procedures.transformers.base import (create_base_args,
                                                        inference_args,
                                                        train_args)
from nlp_architect.models.transformers.question_answering import TransformerQuestionAnswering
from nlp_architect.utils.io import prepare_output_path
from nlp_architect.utils.utils_squad import read_squad_examples

logger = logging.getLogger(__name__)


@register_train_cmd(name='transformer_qa',
                    description='Train a BERT/XLNet model with question answering head')
class TransformerQuestionAnsweringTrain(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--data_dir", default=None, type=str, required=True,
            help="The input data dir. Should contain dataset files to be parsed "
            + "by the dataloaders.")
        parser.add_argument(
            "--max_answer_length", default=30, type=int, help="The maximum length of an "
            + "answer that can be generated. This is needed because the start and end "
            + "predictions are not conditioned on one another.")
        parser.add_argument(
            "--max_query_length", default=64, type=int, help="The maximum number of "
            + "tokens for the question. Questions longer than this will be truncated "
            + "to this length.")
        parser.add_argument(
            "--n_best_size", default=20, type=int, help="The total number of n-best "
            + "predictions to generate in the nbest_predictions.json output file.")
        parser.add_argument(
            "--version_2_with_negative", action='store_true', help="If true, the SQuAD "
            + "examples contain some that do not have an answer.")
        parser.add_argument(
            "--null_score_diff_threshold", type=float, default=0.0, help="If null_score "
            + "- best_non_null is greater than the threshold predict null.")

        train_args(parser, models_family=TransformerQuestionAnswering.MODEL_CLASS.keys())
        create_base_args(parser, model_types=TransformerQuestionAnswering.MODEL_CLASS.keys())

    @staticmethod
    def run_procedure(args):
        do_training(args)


@register_run_cmd(name='transformer_qa',
                  description='Run a BERT/XLNet model with question answering head')
class TransformerQuestionAnsweringRun(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument("--data_file", default=None, type=str, required=True,
                            help="The data file containing data for inference")
        parser.add_argument(
            "--max_answer_length", default=30, type=int, help="The maximum length "
            + "of an answer that can be generated. This is needed because the start "
            + "and end predictions are not conditioned on one another.")
        parser.add_argument(
            "--max_query_length", default=64, type=int, help="The maximum number of "
            + "tokens for the question. Questions longer than this will "
            + "be truncated to this length.")
        parser.add_argument(
            "--n_best_size", default=20, type=int, help="The total number of n-best "
            + "predictions to generate in the nbest_predictions.json output file.")
        parser.add_argument(
            "--version_2_with_negative", action='store_true', help="If true, the SQuAD "
            + 'examples contain some that do not have an answer.')
        parser.add_argument(
            "--null_score_diff_threshold", type=float, default=0.0, help="If null_score - "
            + "best_non_null is greater than the threshold predict null.")
        inference_args(parser)
        create_base_args(parser, model_types=TransformerQuestionAnswering.MODEL_CLASS.keys())

    @staticmethod
    def run_procedure(args):
        do_inference(args)


def do_training(args):
    prepare_output_path(args.output_dir, args.overwrite_output_dir)
    device, n_gpus = setup_backend(args.no_cuda)
    # Set seed
    args.seed = set_seed(args.seed, n_gpus)
    # prepare data
    processor = QuestionAnsweringProcessor(args.data_dir, args.version_2_with_negative)

    classifier = TransformerQuestionAnswering(
        model_type=args.model_type,
        max_answer_length=args.max_answer_length,
        max_query_length=args.max_query_length,
        n_best_size=args.n_best_size,
        version_2_with_negative=args.version_2_with_negative,
        null_score_diff_threshold=args.null_score_diff_threshold,
        model_name_or_path=args.model_name_or_path,
        labels=None,
        config_name=args.config_name,
        tokenizer_name=args.tokenizer_name,
        do_lower_case=args.do_lower_case,
        output_path=args.output_dir,
        device=device,
        n_gpus=n_gpus)

    train_ex = processor.get_train_examples()
    if train_ex is None:
        raise Exception("No train examples found, quitting.")
    dev_ex = processor.get_dev_examples()

    train_batch_size = args.per_gpu_train_batch_size * max(1, n_gpus)

    train_dataset = classifier.convert_to_tensors(
        train_ex, evaluate=False, max_seq_length=args.max_seq_length)
    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(train_dataset, sampler=train_sampler,
                          batch_size=train_batch_size)
    dev_dl = None
    if dev_ex is not None:
        dev_dataset, dev_feat = classifier.convert_to_tensors(
            dev_ex, evaluate=True, max_seq_length=args.max_seq_length)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dl = DataLoader(dev_dataset, sampler=dev_sampler,
                            batch_size=args.per_gpu_eval_batch_size)

    total_steps, _ = classifier.get_train_steps_epochs(args.max_steps,
                                                       args.num_train_epochs,
                                                       args.per_gpu_train_batch_size,
                                                       len(train_dataset))

    classifier.setup_default_optimizer(weight_decay=args.weight_decay,
                                       learning_rate=args.learning_rate,
                                       adam_epsilon=args.adam_epsilon,
                                       warmup_steps=args.warmup_steps,
                                       total_steps=total_steps)
    classifier.train(train_dl, dev_dl, dev_ex, dev_feat,
                     gradient_accumulation_steps=args.gradient_accumulation_steps,
                     per_gpu_train_batch_size=args.per_gpu_train_batch_size,
                     max_steps=args.max_steps,
                     num_train_epochs=args.num_train_epochs,
                     max_grad_norm=args.max_grad_norm,
                     logging_steps=args.logging_steps,
                     save_steps=args.save_steps,
                     data_dir=args.data_dir)
    classifier.save_model(args.output_dir, args=args)


def do_inference(args):
    prepare_output_path(args.output_dir, args.overwrite_output_dir)
    device, n_gpus = setup_backend(args.no_cuda)
    args.batch_size = args.per_gpu_eval_batch_size * max(1, n_gpus)
    inference_examples = process_inference_input(args.data_file, args.version_2_with_negative)
    classifier = TransformerQuestionAnswering.load_model(
        model_path=args.model_path,
        model_type=args.model_type,
        max_answer_length=args.max_answer_length,
        n_best_size=args.n_best_size,
        version_2_with_negative=args.version_2_with_negative,
        null_score_diff_threshold=args.null_score_diff_threshold,
        max_query_length=args.max_query_length,
        output_path=args.output_dir,
        do_lower_case=args.do_lower_case,
        load_quantized=args.load_quantized_model
    )
    classifier.to(device, n_gpus)
    classifier.inference(inference_examples, args.max_seq_length, args.batch_size)


def process_inference_input(input_file, version_2_with_negative):
    if not os.path.exists(input_file):
        logger.error("Requested file %s is not found", input_file)
        return None
    return read_squad_examples(
        input_file, is_training=False,
        version_2_with_negative=version_2_with_negative)
