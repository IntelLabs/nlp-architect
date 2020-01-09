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

from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from nlp_architect.data.glue_tasks import get_glue_task, processors
from nlp_architect.models.transformers import TransformerSequenceClassifier
from nlp_architect.nn.torch import setup_backend, set_seed
from nlp_architect.procedures.procedure import Procedure
from nlp_architect.procedures.registry import register_inference_cmd, register_train_cmd
from nlp_architect.procedures.transformers.base import create_base_args, inference_args, train_args
from nlp_architect.utils.io import prepare_output_path
from nlp_architect.utils.metrics import acc_and_f1, pearson_and_spearman, simple_accuracy

logger = logging.getLogger(__name__)


@register_train_cmd(
    name="transformer_glue", description="Train (finetune) a BERT/XLNet/XLM model on a GLUE task"
)
class TransformerGlueTrain(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add_glue_args(parser)
        create_base_args(parser, model_types=TransformerSequenceClassifier.MODEL_CLASS.keys())
        train_args(parser, models_family=TransformerSequenceClassifier.MODEL_CLASS.keys())

    @staticmethod
    def run_procedure(args):
        do_training(args)


@register_inference_cmd(
    name="transformer_glue", description="Run a BERT/XLNet/XLM model on a GLUE task"
)
class TransformerGlueRun(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add_glue_args(parser)
        add_glue_inference_args(parser)
        inference_args(parser)
        create_base_args(parser, model_types=TransformerSequenceClassifier.MODEL_CLASS.keys())

    @staticmethod
    def run_procedure(args):
        do_inference(args)


def add_glue_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain dataset files to be parsed "
        + "by the dataloaders.",
    )


def add_glue_inference_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate the model on the task's development set"
    )


def do_training(args):
    prepare_output_path(args.output_dir, args.overwrite_output_dir)
    device, n_gpus = setup_backend(args.no_cuda)
    # Set seed
    args.seed = set_seed(args.seed, n_gpus)
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    task = get_glue_task(args.task_name, data_dir=args.data_dir)
    classifier = TransformerSequenceClassifier(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        labels=task.get_labels(),
        task_type=task.task_type,
        metric_fn=get_metric_fn(task.name),
        config_name=args.config_name,
        tokenizer_name=args.tokenizer_name,
        do_lower_case=args.do_lower_case,
        output_path=args.output_dir,
        device=device,
        n_gpus=n_gpus,
    )

    train_batch_size = args.per_gpu_train_batch_size * max(1, n_gpus)

    train_ex = task.get_train_examples()
    dev_ex = task.get_dev_examples()
    train_dataset = classifier.convert_to_tensors(train_ex, args.max_seq_length)
    dev_dataset = classifier.convert_to_tensors(dev_ex, args.max_seq_length)
    train_sampler = RandomSampler(train_dataset)
    dev_sampler = SequentialSampler(dev_dataset)
    train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    dev_dl = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.per_gpu_eval_batch_size)

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
        None,
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
    args.task_name = args.task_name.lower()
    task = get_glue_task(args.task_name, data_dir=args.data_dir)
    args.batch_size = args.per_gpu_eval_batch_size * max(1, n_gpus)
    classifier = TransformerSequenceClassifier.load_model(
        model_path=args.model_path,
        model_type=args.model_type,
        task_type=task.task_type,
        metric_fn=get_metric_fn(task.name),
        do_lower_case=args.do_lower_case,
        load_quantized=args.load_quantized_model,
    )
    classifier.to(device, n_gpus)
    examples = task.get_dev_examples() if args.evaluate else task.get_test_examples()
    preds = classifier.inference(
        examples, args.max_seq_length, args.batch_size, evaluate=args.evaluate
    )
    with io.open(os.path.join(args.output_dir, "output.txt"), "w", encoding="utf-8") as fw:
        for p in preds:
            fw.write("{}\n".format(p))


# GLUE task metrics
def get_metric_fn(task_name):
    if task_name == "cola":
        return lambda p, l: {"mcc": matthews_corrcoef(p, l)}
    if task_name == "sst-2":
        return lambda p, l: {"acc": simple_accuracy(p, l)}
    if task_name == "mrpc":
        return acc_and_f1
    if task_name == "sts-b":
        return pearson_and_spearman
    if task_name == "qqp":
        return acc_and_f1
    if task_name == "mnli":
        return lambda p, l: {"acc": simple_accuracy(p, l)}
    if task_name == "mnli-mm":
        return lambda p, l: {"acc": simple_accuracy(p, l)}
    if task_name == "qnli":
        return lambda p, l: {"acc": simple_accuracy(p, l)}
    if task_name == "rte":
        return lambda p, l: {"acc": simple_accuracy(p, l)}
    if task_name == "wnli":
        return lambda p, l: {"acc": simple_accuracy(p, l)}
    raise KeyError(task_name)
