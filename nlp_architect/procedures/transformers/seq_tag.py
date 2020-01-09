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
import io
import logging
import os

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from nlp_architect.data.sequential_tagging import TokenClsInputExample, TokenClsProcessor
from nlp_architect.data.utils import write_column_tagged_file
from nlp_architect.models.transformers import TransformerTokenClassifier
from nlp_architect.nn.torch import setup_backend, set_seed
from nlp_architect.procedures.procedure import Procedure
from nlp_architect.procedures.registry import register_inference_cmd, register_train_cmd
from nlp_architect.procedures.transformers.base import create_base_args, inference_args, train_args
from nlp_architect.utils.io import prepare_output_path
from nlp_architect.utils.text import SpacyInstance

logger = logging.getLogger(__name__)


@register_train_cmd(
    name="transformer_token", description="Train a BERT/XLNet model with token classification head"
)
class TransformerTokenClsTrain(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain dataset files to be parsed "
            + "by the dataloaders.",
        )
        train_args(parser, models_family=TransformerTokenClassifier.MODEL_CLASS.keys())
        create_base_args(parser, model_types=TransformerTokenClassifier.MODEL_CLASS.keys())
        parser.add_argument(
            "--train_file_name",
            type=str,
            default="train.txt",
            help="File name of the training dataset",
        )

    @staticmethod
    def run_procedure(args):
        do_training(args)


@register_inference_cmd(
    name="transformer_token", description="Run a BERT/XLNet model with token classification head"
)
class TransformerTokenClsRun(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--data_file",
            default=None,
            type=str,
            required=True,
            help="The data file containing data for inference",
        )
        inference_args(parser)
        create_base_args(parser, model_types=TransformerTokenClassifier.MODEL_CLASS.keys())

    @staticmethod
    def run_procedure(args):
        do_inference(args)


def do_training(args):
    prepare_output_path(args.output_dir, args.overwrite_output_dir)
    device, n_gpus = setup_backend(args.no_cuda)
    # Set seed
    args.seed = set_seed(args.seed, n_gpus)
    # prepare data
    processor = TokenClsProcessor(args.data_dir)

    classifier = TransformerTokenClassifier(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        labels=processor.get_labels(),
        config_name=args.config_name,
        tokenizer_name=args.tokenizer_name,
        do_lower_case=args.do_lower_case,
        output_path=args.output_dir,
        device=device,
        n_gpus=n_gpus,
    )

    train_ex = processor.get_train_examples(filename=args.train_file_name)
    if train_ex is None:
        raise Exception("No train examples found, quitting.")
    dev_ex = processor.get_dev_examples()
    test_ex = processor.get_test_examples()

    train_batch_size = args.per_gpu_train_batch_size * max(1, n_gpus)

    train_dataset = classifier.convert_to_tensors(train_ex, max_seq_length=args.max_seq_length)
    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    dev_dl = None
    test_dl = None
    if dev_ex is not None:
        dev_dataset = classifier.convert_to_tensors(dev_ex, max_seq_length=args.max_seq_length)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dl = DataLoader(
            dev_dataset, sampler=dev_sampler, batch_size=args.per_gpu_eval_batch_size
        )

    if test_ex is not None:
        test_dataset = classifier.convert_to_tensors(test_ex, max_seq_length=args.max_seq_length)
        test_sampler = SequentialSampler(test_dataset)
        test_dl = DataLoader(
            test_dataset, sampler=test_sampler, batch_size=args.per_gpu_eval_batch_size
        )

    total_steps, _ = classifier.get_train_steps_epochs(
        args.max_steps, args.num_train_epochs, args.per_gpu_train_batch_size, len(train_dataset)
    )

    classifier.setup_default_optimizer(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )
    classifier.train(
        train_dl,
        dev_dl,
        test_dl,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_gpu_train_batch_size=args.per_gpu_train_batch_size,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )
    classifier.save_model(args.output_dir, args=args)


def do_inference(args):
    prepare_output_path(args.output_dir, args.overwrite_output_dir)
    device, n_gpus = setup_backend(args.no_cuda)
    args.batch_size = args.per_gpu_eval_batch_size * max(1, n_gpus)
    inference_examples = process_inference_input(args.data_file)
    classifier = TransformerTokenClassifier.load_model(
        model_path=args.model_path,
        model_type=args.model_type,
        do_lower_case=args.do_lower_case,
        load_quantized=args.load_quantized_model,
    )
    classifier.to(device, n_gpus)
    output = classifier.inference(inference_examples, args.max_seq_length, args.batch_size)
    write_column_tagged_file(args.output_dir + os.sep + "output.txt", output)


def process_inference_input(input_file):
    with io.open(input_file) as fp:
        texts = [l.strip() for l in fp.readlines()]
    tokenizer = SpacyInstance(disable=["tagger", "parser", "ner"])
    examples = []
    for i, t in enumerate(texts):
        examples.append(TokenClsInputExample(str(i), t, tokenizer.tokenize(t)))
    return examples
