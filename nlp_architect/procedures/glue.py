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
import torch

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
from nlp_architect.models.glue import GLUE, CNN
from nlp_architect.utils.embedding import get_embedding_matrix, load_embedding_file
from nlp_architect.nn.torch.distillation import TeacherStudentDistill
from nlp_architect.nn.torch.data.dataset import ParallelDataset, ConcatTensorDataset, CombinedTensorDataset

logger = logging.getLogger(__name__)


@register_train_cmd(
    name="glue", description="Train a classifier on a GLUE task"
)
class TrainGlue(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add_parse_args(parser)

    @staticmethod
    def run_procedure(args):
        do_training(args)


@register_train_cmd(
    name="glue_kd",
    description="Train a classifier on a GLUE task using Knowledge Distillation"
    " and a Transformer teacher model",
)

class TrainGlueKD(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add_parse_args(parser)
        TeacherStudentDistill.add_args(parser)
        parser.add_argument("--teacher_max_seq_len", type=int, default=128, help="Max sentence \
                             length for teacher data loading")

    @staticmethod
    def run_procedure(args):
        do_kd_training(args)


@register_train_cmd(
    name="glue_kd_pseudo",
    description="Train a classifier on a GLUE task using Knowledge Distillation"
    " and a Transformer teacher model + pseudo-labeling",
)
class TrainGlueKDPseudo(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add_parse_args(parser)
        TeacherStudentDistill.add_args(parser)
        parser.add_argument("--unlabeled_filename", default="unlabeled.txt", type=str,
                            help="The file name containing the unlabeled training examples")
        parser.add_argument('--parallel_batching', action='store_true',
                            help="sample labeled/unlabeled batch in parallel")
        parser.add_argument("--teacher_max_seq_len", type=int, default=128, help="Max sentence \
                             length for teacher data loading")


    @staticmethod
    def run_procedure(args):
        do_kd_pseudo_training(args)


@register_inference_cmd(
    name="glue", description="Run a classifier on a GLUE task"
)
class GlueRun(Procedure):
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

def add_parse_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model_type",
        default="cnn-lstm",
        type=str,
        choices=list(MODEL_TYPE.keys()),
        help="model type to use for this tagger",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument("--config_file", type=str, help="Embedder model configuration file")
    parser.add_argument("-b", type=int, default=10, help="Batch size")
    parser.add_argument("-e", type=int, default=155, help="Number of epochs")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain dataset files to be parsed by " "the dataloaders.",
    )
    parser.add_argument(
        "--tag_col", type=int, default=-1, help="Entity labels tab number in train/test files"
    )
    parser.add_argument("--max_sentence_length", type=int, default=50, help="Max sentence length")
    parser.add_argument(
        "--max_word_length", type=int, default=12, help="Max word length in characters"
    )
    parser.add_argument(
        "--use_crf", action="store_true", help="Use CRF classifier instead of Softmax"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for optimizer (Adam)"
    )
    parser.add_argument("--embedding_file", help="Path to external word embedding model file")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory where the model will be saved",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save model every X updates steps."
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--best_result_file",
        type=str,
        default="best_dev.txt",
        help="file path for best evaluation output",
    )
    parser.add_argument(
        "--word_dropout", type=float, default=0, help="word dropout rate for input tokens"
    )
    parser.add_argument(
        "--ignore_token", type=str, default="", help="a token to ignore when processing the data"
    )
    parser.add_argument(
        "--train_filename", type=str, default="train.txt", help="filename of training dataset"
    )
    parser.add_argument(
        "--dev_filename", type=str, default="dev.txt", help="filename of development dataset"
    )
    parser.add_argument(
        "--test_filename", type=str, default="test.txt", help="filename of test dataset"
    )



MODEL_TYPE = {"cola": CNN,
              "sst-2": CNN,
              }

def do_training(args):
    prepare_output_path(args.output_dir, args.overwrite_output_dir)
    device, n_gpus = setup_backend(args.no_cuda)
    # Set seed
    args.seed = set_seed(args.seed, n_gpus)
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    task = get_glue_task(args.task_name, data_dir=args.data_dir)
    train_ex = task.get_train_examples(filename=args.train_filename)
    dev_ex = task.get_dev_examples()
    vocab = task.get_vocabulary(train_ex + dev_ex)
    vocab_size = len(vocab) + 1
    num_labels = len(task.get_labels())

    # create an embedder
    embedder_cls = MODEL_TYPE[args.task_name]
    if args.config_file is not None:
        embedder_model = embedder_cls.from_config(vocab_size, num_labels, args.config_file)
    else:
        embedder_model = embedder_cls(vocab_size, num_labels, max_seq_len=args.max_sentence_length)

    # load external word embeddings if present
    if args.embedding_file is not None:
        emb_dict = load_embedding_file(args.embedding_file, dim=embedder_model.embedding_dim)
        emb_mat = get_embedding_matrix(emb_dict, vocab)
        emb_mat = torch.tensor(emb_mat, dtype=torch.float)
        embedder_model.load_embeddings(emb_mat)

    classifier = GLUE(
        embedder_model,
        word_vocab=vocab,
        labels=task.get_labels(),
        device=device,
        n_gpus=n_gpus,
        metric_fn=get_metric_fn(task.name)
    )

    train_batch_size = args.b * max(1, n_gpus)

    train_dataset = classifier.convert_to_tensors(train_ex, max_seq_length=args.max_sentence_length)
    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    if dev_ex is not None:
        dev_dataset = classifier.convert_to_tensors(dev_ex, max_seq_length=args.max_sentence_length)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dl = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.b)

    if args.lr is not None:
        opt = classifier.get_optimizer(lr=args.lr)
    classifier.train(
        train_dl,
        dev_dl,
        epochs=args.e,
        batch_size=args.b,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_path=args.output_dir,
        optimizer=opt if opt is not None else None,
        best_result_file=args.best_result_file,
        word_dropout=args.word_dropout,
    )
    classifier.save_model(args.output_dir)



def do_kd_training(args):
    prepare_output_path(args.output_dir, args.overwrite_output_dir)
    device, n_gpus = setup_backend(args.no_cuda)
    # Set seed
    args.seed = set_seed(args.seed, n_gpus)
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    task = get_glue_task(args.task_name, data_dir=args.data_dir)
    train_ex = task.get_train_examples(filename=args.train_filename)
    dev_ex = task.get_dev_examples()
    vocab = task.get_vocabulary(train_ex + dev_ex)
    vocab_size = len(vocab) + 1
    num_labels = len(task.get_labels())

    # create an embedder
    embedder_cls = MODEL_TYPE[args.task_name]
    if args.config_file is not None:
        embedder_model = embedder_cls.from_config(vocab_size, num_labels, args.config_file)
    else:
        embedder_model = embedder_cls(vocab_size, num_labels, max_seq_len=args.max_sentence_length)

    # load external word embeddings if present
    if args.embedding_file is not None:
        emb_dict = load_embedding_file(args.embedding_file, dim=embedder_model.embedding_dim)
        emb_mat = get_embedding_matrix(emb_dict, vocab)
        emb_mat = torch.tensor(emb_mat, dtype=torch.float)
        embedder_model.load_embeddings(emb_mat)

    classifier = GLUE(
        embedder_model,
        word_vocab=vocab,
        labels=task.get_labels(),
        device=device,
        n_gpus=n_gpus,
        metric_fn=get_metric_fn(task.name)
    )

    train_batch_size = args.b * max(1, n_gpus)

    train_dataset = classifier.convert_to_tensors(train_ex, max_seq_length=args.max_sentence_length)

    # load saved teacher args if exist
    if os.path.exists(args.teacher_model_path + os.sep + "training_args.bin"):
        t_args = torch.load(args.teacher_model_path + os.sep + "training_args.bin")
        t_device, t_n_gpus = setup_backend(t_args.no_cuda)
        teacher = TransformerSequenceClassifier.load_model(
            model_type=args.model_type,
            model_path=args.model_name_or_path,
            labels=task.get_labels(),
            task_type=task.task_type,
            metric_fn=get_metric_fn(task.name),
            config_name=t_args.config_name,
            tokenizer_name=t_args.tokenizer_name,
            do_lower_case=t_args.do_lower_case,
            output_path=t_args.output_dir,
            device=t_device,
            n_gpus=t_n_gpus,
        )

    else:
        teacher = TransformerSequenceClassifier.load_model(
            model_path=args.teacher_model_path, model_type=args.teacher_model_type
        )
        teacher.to(device, n_gpus)

    teacher_dataset = teacher.convert_to_tensors(
        train_ex, max_seq_length=args.teacher_max_seq_len, include_labels=False
    )

    train_dataset = ParallelDataset(train_dataset, teacher_dataset)

    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    if dev_ex is not None:
        dev_dataset = classifier.convert_to_tensors(dev_ex, max_seq_length=args.max_sentence_length)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dl = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.b)

    if args.lr is not None:
        opt = classifier.get_optimizer(lr=args.lr)


    distiller = TeacherStudentDistill(
            teacher, args.kd_temp, args.kd_dist_w, args.kd_student_w, args.kd_loss_fn
        )

    classifier.train(
        train_dl,
        dev_dl,
        epochs=args.e,
        batch_size=args.b,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_path=args.output_dir,
        optimizer=opt if opt is not None else None,
        distiller=distiller,
        best_result_file=args.best_result_file,
        word_dropout=args.word_dropout,
    )
    classifier.save_model(args.output_dir)


def do_kd_pseudo_training(args):
    prepare_output_path(args.output_dir, args.overwrite_output_dir)
    device, n_gpus = setup_backend(args.no_cuda)
    # Set seed
    args.seed = set_seed(args.seed, n_gpus)
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    task = get_glue_task(args.task_name, data_dir=args.data_dir)
    
    

    train_labeled_ex = task.get_train_examples(filename=args.train_filename)
    train_unlabeled_ex = task.get_train_examples(filename=args.unlabeled_filename)

    dev_ex = task.get_dev_examples()
    vocab = task.get_vocabulary(train_labeled_ex + train_unlabeled_ex + dev_ex)

    vocab_size = len(vocab) + 1
    num_labels = len(task.get_labels())

    # create an embedder
    embedder_cls = MODEL_TYPE[args.task_name]
    if args.config_file is not None:
        embedder_model = embedder_cls.from_config(vocab_size, num_labels, args.config_file)
    else:
        embedder_model = embedder_cls(vocab_size, num_labels, max_seq_len=args.max_sentence_length)

    # load external word embeddings if present
    if args.embedding_file is not None:
        emb_dict = load_embedding_file(args.embedding_file, dim=embedder_model.embedding_dim)
        emb_mat = get_embedding_matrix(emb_dict, vocab)
        emb_mat = torch.tensor(emb_mat, dtype=torch.float)
        embedder_model.load_embeddings(emb_mat)

    classifier = GLUE(
        embedder_model,
        word_vocab=vocab,
        labels=task.get_labels(),
        device=device,
        n_gpus=n_gpus,
        metric_fn=get_metric_fn(task.name)
    )

    train_batch_size = args.b * max(1, n_gpus)

    train_labeled_dataset = classifier.convert_to_tensors(
        train_labeled_ex,
        max_seq_length=args.max_sentence_length
    )
    train_unlabeled_dataset = classifier.convert_to_tensors(
        train_unlabeled_ex, max_seq_length=args.max_sentence_length,
        include_labels=False)

    train_dataset = CombinedTensorDataset(train_labeled_dataset, train_unlabeled_dataset)

    # load saved teacher args if exist
    if os.path.exists(args.teacher_model_path + os.sep + "training_args.bin"):
        t_args = torch.load(args.teacher_model_path + os.sep + "training_args.bin")
        t_device, t_n_gpus = setup_backend(t_args.no_cuda)
        teacher = TransformerSequenceClassifier.load_model(
            model_type=args.model_type,
            model_path=args.model_name_or_path,
            labels=task.get_labels(),
            task_type=task.task_type,
            metric_fn=get_metric_fn(task.name),
            config_name=t_args.config_name,
            tokenizer_name=t_args.tokenizer_name,
            do_lower_case=t_args.do_lower_case,
            output_path=t_args.output_dir,
            device=t_device,
            n_gpus=t_n_gpus,
        )

    else:
        teacher = TransformerSequenceClassifier.load_model(
            model_path=args.teacher_model_path, model_type=args.teacher_model_type
        )
        teacher.to(device, n_gpus)

    teacher_labeled_dataset = teacher.convert_to_tensors(
        train_labeled_ex, args.teacher_max_seq_len)
    teacher_unlabeled_dataset = teacher.convert_to_tensors(
        train_unlabeled_ex, args.teacher_max_seq_len, False
    )
    teacher_dataset = CombinedTensorDataset(teacher_labeled_dataset, teacher_unlabeled_dataset)

    train_dataset = ParallelDataset(train_dataset, teacher_dataset)
    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=train_batch_size)



    if dev_ex is not None:
        dev_dataset = classifier.convert_to_tensors(dev_ex, max_seq_length=args.max_sentence_length)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dl = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.b)

    if args.lr is not None:
        opt = classifier.get_optimizer(lr=args.lr)


    distiller = TeacherStudentDistill(
            teacher, args.kd_temp, args.kd_dist_w, args.kd_student_w, args.kd_loss_fn
        )

    classifier.train(
        train_dl,
        dev_dl,
        epochs=args.e,
        batch_size=args.b,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_path=args.output_dir,
        optimizer=opt if opt is not None else None,
        distiller=distiller,
        best_result_file=args.best_result_file,
        word_dropout=args.word_dropout,
    )
    classifier.save_model(args.output_dir)


def do_inference(args):  # TODO
    print("Not implemented")


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
