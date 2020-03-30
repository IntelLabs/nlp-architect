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
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from nlp_architect.data.sequential_tagging import TokenClsInputExample, TokenClsProcessor
from nlp_architect.data.utils import write_column_tagged_file
from nlp_architect.models.tagging import NeuralTagger
from nlp_architect.nn.torch.modules.embedders import IDCNN, CNNLSTM
from nlp_architect.nn.torch import setup_backend, set_seed
from nlp_architect.nn.torch.distillation import TeacherStudentDistill
from nlp_architect.procedures.procedure import Procedure
from nlp_architect.procedures.registry import register_train_cmd, register_inference_cmd
from nlp_architect.utils.embedding import get_embedding_matrix, load_embedding_file
from nlp_architect.utils.io import prepare_output_path
from nlp_architect.utils.text import SpacyInstance
from nlp_architect.nn.torch.data.dataset import (
    ParallelDataset,
    ConcatTensorDataset,
    CombinedTensorDataset,
)
from nlp_architect.models.transformers import TransformerTokenClassifier


@register_train_cmd(name="tagger", description="Train a neural tagger")
class TrainTagger(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add_parse_args(parser)

    @staticmethod
    def run_procedure(args):
        do_training(args)


@register_train_cmd(
    name="tagger_kd",
    description="Train a neural tagger using Knowledge Distillation"
    " and a Transformer teacher model",
)
class TrainTaggerKD(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add_parse_args(parser)
        TeacherStudentDistill.add_args(parser)
        parser.add_argument(
            "--teacher_max_seq_len",
            type=int,
            default=128,
            help="Max sentence \
                             length for teacher data loading",
        )

    @staticmethod
    def run_procedure(args):
        do_kd_training(args)


@register_train_cmd(
    name="tagger_kd_pseudo",
    description="Train a neural tagger using Knowledge Distillation"
    " and a Transformer teacher model + pseudo-labeling",
)
class TrainTaggerKDPseudo(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        add_parse_args(parser)
        TeacherStudentDistill.add_args(parser)
        parser.add_argument(
            "--unlabeled_filename",
            default="unlabeled.txt",
            type=str,
            help="The file name containing the unlabeled training examples",
        )
        parser.add_argument(
            "--parallel_batching",
            action="store_true",
            help="sample labeled/unlabeled batch in parallel",
        )
        parser.add_argument(
            "--teacher_max_seq_len",
            type=int,
            default=128,
            help="Max sentence \
                             length for teacher data loading",
        )

    @staticmethod
    def run_procedure(args):
        do_kd_pseudo_training(args)


@register_inference_cmd(name="tagger", description="Run a neural tagger model")
class RunTagger(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--data_file",
            default=None,
            type=str,
            required=True,
            help="The data file containing data for inference",
        )
        parser.add_argument(
            "--model_dir", type=str, required=True, help="Path to trained model directory"
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            required=True,
            help="Output directory where the model will be saved",
        )
        parser.add_argument(
            "--overwrite_output_dir",
            action="store_true",
            help="Overwrite the content of the output directory",
        )
        parser.add_argument(
            "--no_cuda", action="store_true", help="Avoid using CUDA when available"
        )
        parser.add_argument("-b", type=int, default=100, help="Batch size")

    @staticmethod
    def run_procedure(args):
        do_inference(args)


def add_parse_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model_type",
        default="cnn-lstm",
        type=str,
        choices=list(MODEL_TYPE.keys()),
        help="model type to use for this tagger",
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


MODEL_TYPE = {"cnn-lstm": CNNLSTM, "id-cnn": IDCNN}


def do_training(args):
    prepare_output_path(args.output_dir, args.overwrite_output_dir)
    device, n_gpus = setup_backend(args.no_cuda)
    # Set seed
    args.seed = set_seed(args.seed, n_gpus)
    # prepare data
    processor = TokenClsProcessor(
        args.data_dir, tag_col=args.tag_col, ignore_token=args.ignore_token
    )
    train_ex = processor.get_train_examples(filename=args.train_filename)
    dev_ex = processor.get_dev_examples(filename=args.dev_filename)
    test_ex = processor.get_test_examples(filename=args.test_filename)
    vocab = processor.get_vocabulary(train_ex + dev_ex + test_ex)
    vocab_size = len(vocab) + 1
    num_labels = len(processor.get_labels()) + 1
    # create an embedder
    embedder_cls = MODEL_TYPE[args.model_type]
    if args.config_file is not None:
        embedder_model = embedder_cls.from_config(vocab_size, num_labels, args.config_file)
    else:
        embedder_model = embedder_cls(vocab_size, num_labels)

    # load external word embeddings if present
    if args.embedding_file is not None:
        emb_dict = load_embedding_file(args.embedding_file, dim=embedder_model.word_embedding_dim)
        emb_mat = get_embedding_matrix(emb_dict, vocab)
        emb_mat = torch.tensor(emb_mat, dtype=torch.float)
        embedder_model.load_embeddings(emb_mat)

    classifier = NeuralTagger(
        embedder_model,
        word_vocab=vocab,
        labels=processor.get_labels(),
        use_crf=args.use_crf,
        device=device,
        n_gpus=n_gpus,
    )

    train_batch_size = args.b * max(1, n_gpus)

    train_dataset = classifier.convert_to_tensors(
        train_ex, max_seq_length=args.max_sentence_length, max_word_length=args.max_word_length
    )
    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    if dev_ex is not None:
        dev_dataset = classifier.convert_to_tensors(
            dev_ex, max_seq_length=args.max_sentence_length, max_word_length=args.max_word_length
        )
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dl = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.b)

    if test_ex is not None:
        test_dataset = classifier.convert_to_tensors(
            test_ex, max_seq_length=args.max_sentence_length, max_word_length=args.max_word_length
        )
        test_sampler = SequentialSampler(test_dataset)
        test_dl = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.b)
    if args.lr is not None:
        opt = classifier.get_optimizer(lr=args.lr)
    classifier.train(
        train_dl,
        dev_dl,
        test_dl,
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
    # prepare data
    processor = TokenClsProcessor(
        args.data_dir, tag_col=args.tag_col, ignore_token=args.ignore_token
    )
    train_ex = processor.get_train_examples(filename=args.train_filename)
    dev_ex = processor.get_dev_examples(filename=args.dev_filename)
    test_ex = processor.get_test_examples(filename=args.test_filename)
    vocab = processor.get_vocabulary(train_ex + dev_ex + test_ex)
    vocab_size = len(vocab) + 1
    num_labels = len(processor.get_labels()) + 1
    # create an embedder
    embedder_cls = MODEL_TYPE[args.model_type]
    if args.config_file is not None:
        embedder_model = embedder_cls.from_config(vocab_size, num_labels, args.config_file)
    else:
        embedder_model = embedder_cls(vocab_size, num_labels)

    # load external word embeddings if present
    if args.embedding_file is not None:
        emb_dict = load_embedding_file(args.embedding_file, dim=embedder_model.word_embedding_dim)
        emb_mat = get_embedding_matrix(emb_dict, vocab)
        emb_mat = torch.tensor(emb_mat, dtype=torch.float)
        embedder_model.load_embeddings(emb_mat)

    classifier = NeuralTagger(
        embedder_model,
        word_vocab=vocab,
        labels=processor.get_labels(),
        use_crf=args.use_crf,
        device=device,
        n_gpus=n_gpus,
    )

    train_batch_size = args.b * max(1, n_gpus)
    train_dataset = classifier.convert_to_tensors(
        train_ex, max_seq_length=args.max_sentence_length, max_word_length=args.max_word_length
    )

    # load saved teacher args if exist
    if os.path.exists(args.teacher_model_path + os.sep + "training_args.bin"):
        t_args = torch.load(args.teacher_model_path + os.sep + "training_args.bin")
        t_device, t_n_gpus = setup_backend(t_args.no_cuda)
        teacher = TransformerTokenClassifier.load_model(
            model_path=args.teacher_model_path,
            model_type=args.teacher_model_type,
            config_name=t_args.config_name,
            tokenizer_name=t_args.tokenizer_name,
            do_lower_case=t_args.do_lower_case,
            output_path=t_args.output_dir,
            device=t_device,
            n_gpus=t_n_gpus,
        )
    else:
        teacher = TransformerTokenClassifier.load_model(
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
        dev_dataset = classifier.convert_to_tensors(
            dev_ex, max_seq_length=args.max_sentence_length, max_word_length=args.max_word_length
        )
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dl = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.b)

    if test_ex is not None:
        test_dataset = classifier.convert_to_tensors(
            test_ex, max_seq_length=args.max_sentence_length, max_word_length=args.max_word_length
        )
        test_sampler = SequentialSampler(test_dataset)
        test_dl = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.b)
    if args.lr is not None:
        opt = classifier.get_optimizer(lr=args.lr)

    distiller = TeacherStudentDistill(
        teacher, args.kd_temp, args.kd_dist_w, args.kd_student_w, args.kd_loss_fn
    )
    classifier.train(
        train_dl,
        dev_dl,
        test_dl,
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
    # prepare data
    processor = TokenClsProcessor(
        args.data_dir, tag_col=args.tag_col, ignore_token=args.ignore_token
    )
    train_labeled_ex = processor.get_train_examples(filename=args.train_filename)
    train_unlabeled_ex = processor.get_train_examples(filename=args.unlabeled_filename)
    dev_ex = processor.get_dev_examples(filename=args.dev_filename)
    test_ex = processor.get_test_examples(filename=args.test_filename)
    vocab = processor.get_vocabulary(train_labeled_ex + train_unlabeled_ex + dev_ex + test_ex)
    vocab_size = len(vocab) + 1
    num_labels = len(processor.get_labels()) + 1
    # create an embedder
    embedder_cls = MODEL_TYPE[args.model_type]
    if args.config_file is not None:
        embedder_model = embedder_cls.from_config(vocab_size, num_labels, args.config_file)
    else:
        embedder_model = embedder_cls(vocab_size, num_labels)

    # load external word embeddings if present
    if args.embedding_file is not None:
        emb_dict = load_embedding_file(args.embedding_file, dim=embedder_model.word_embedding_dim)
        emb_mat = get_embedding_matrix(emb_dict, vocab)
        emb_mat = torch.tensor(emb_mat, dtype=torch.float)
        embedder_model.load_embeddings(emb_mat)

    classifier = NeuralTagger(
        embedder_model,
        word_vocab=vocab,
        labels=processor.get_labels(),
        use_crf=args.use_crf,
        device=device,
        n_gpus=n_gpus,
    )

    train_batch_size = args.b * max(1, n_gpus)
    train_labeled_dataset = classifier.convert_to_tensors(
        train_labeled_ex,
        max_seq_length=args.max_sentence_length,
        max_word_length=args.max_word_length,
    )
    train_unlabeled_dataset = classifier.convert_to_tensors(
        train_unlabeled_ex,
        max_seq_length=args.max_sentence_length,
        max_word_length=args.max_word_length,
        include_labels=False,
    )

    if args.parallel_batching:
        # # concat labeled+unlabeled dataset
        # train_dataset = ConcatTensorDataset(train_labeled_dataset, [train_unlabeled_dataset])
        # match sizes of labeled/unlabeled train data for parallel batching
        larger_ds, smaller_ds = (
            (train_labeled_dataset, train_unlabeled_dataset)
            if len(train_labeled_dataset) > len(train_unlabeled_dataset)
            else (train_unlabeled_dataset, train_labeled_dataset)
        )
        concat_smaller_ds = smaller_ds
        while len(concat_smaller_ds) < len(larger_ds):
            concat_smaller_ds = ConcatTensorDataset(concat_smaller_ds, [smaller_ds])
        if len(concat_smaller_ds[0]) == 4:
            train_unlabeled_dataset = concat_smaller_ds
        else:
            train_labeled_dataset = concat_smaller_ds
    else:
        train_dataset = CombinedTensorDataset([train_labeled_dataset, train_unlabeled_dataset])

    # load saved teacher args if exist
    if os.path.exists(args.teacher_model_path + os.sep + "training_args.bin"):
        t_args = torch.load(args.teacher_model_path + os.sep + "training_args.bin")
        t_device, t_n_gpus = setup_backend(t_args.no_cuda)
        teacher = TransformerTokenClassifier.load_model(
            model_path=args.teacher_model_path,
            model_type=args.teacher_model_type,
            config_name=t_args.config_name,
            tokenizer_name=t_args.tokenizer_name,
            do_lower_case=t_args.do_lower_case,
            output_path=t_args.output_dir,
            device=t_device,
            n_gpus=t_n_gpus,
        )
    else:
        teacher = TransformerTokenClassifier.load_model(
            model_path=args.teacher_model_path, model_type=args.teacher_model_type
        )
        teacher.to(device, n_gpus)

    teacher_labeled_dataset = teacher.convert_to_tensors(train_labeled_ex, args.teacher_max_seq_len)
    teacher_unlabeled_dataset = teacher.convert_to_tensors(
        train_unlabeled_ex, args.teacher_max_seq_len, False
    )

    if args.parallel_batching:
        # # concat teacher labeled+unlabeled dataset
        # teacher_dataset = ConcatTensorDataset(teacher_labeled_dataset, [teacher_unlabeled_dataset])
        # match sizes of labeled/unlabeled teacher train data for parallel batching
        larger_ds, smaller_ds = (
            (teacher_labeled_dataset, teacher_unlabeled_dataset)
            if len(teacher_labeled_dataset) > len(teacher_unlabeled_dataset)
            else (teacher_unlabeled_dataset, teacher_labeled_dataset)
        )
        concat_smaller_ds = smaller_ds
        while len(concat_smaller_ds) < len(larger_ds):
            concat_smaller_ds = ConcatTensorDataset(concat_smaller_ds, [smaller_ds])
        if len(concat_smaller_ds[0]) == 4:
            teacher_unlabeled_dataset = concat_smaller_ds
        else:
            teacher_labeled_dataset = concat_smaller_ds

        train_all_dataset = ParallelDataset(
            train_labeled_dataset,
            teacher_labeled_dataset,
            train_unlabeled_dataset,
            teacher_unlabeled_dataset,
        )

        train_all_sampler = RandomSampler(train_all_dataset)
        # this way must use same batch size for both labeled/unlabeled sets
        train_dl = DataLoader(
            train_all_dataset, sampler=train_all_sampler, batch_size=train_batch_size
        )

    else:
        teacher_dataset = CombinedTensorDataset(
            [teacher_labeled_dataset, teacher_unlabeled_dataset]
        )

        train_dataset = ParallelDataset(train_dataset, teacher_dataset)
        train_sampler = RandomSampler(train_dataset)
        train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    if dev_ex is not None:
        dev_dataset = classifier.convert_to_tensors(
            dev_ex, max_seq_length=args.max_sentence_length, max_word_length=args.max_word_length
        )
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dl = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.b)

    if test_ex is not None:
        test_dataset = classifier.convert_to_tensors(
            test_ex, max_seq_length=args.max_sentence_length, max_word_length=args.max_word_length
        )
        test_sampler = SequentialSampler(test_dataset)
        test_dl = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.b)
    if args.lr is not None:
        opt = classifier.get_optimizer(lr=args.lr)

    distiller = TeacherStudentDistill(
        teacher, args.kd_temp, args.kd_dist_w, args.kd_student_w, args.kd_loss_fn
    )

    classifier.train(
        train_dl,
        dev_dl,
        test_dl,
        epochs=args.e,
        batch_size=args.b,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_path=args.output_dir,
        optimizer=opt if opt is not None else None,
        best_result_file=args.best_result_file,
        distiller=distiller,
        word_dropout=args.word_dropout,
    )

    classifier.save_model(args.output_dir)


def do_inference(args):
    prepare_output_path(args.output_dir, args.overwrite_output_dir)
    device, n_gpus = setup_backend(args.no_cuda)
    args.batch_size = args.b * max(1, n_gpus)
    inference_examples = process_inference_input(args.data_file)
    classifier = NeuralTagger.load_model(model_path=args.model_dir)
    classifier.to(device, n_gpus)
    output = classifier.inference(inference_examples, args.b)
    write_column_tagged_file(args.output_dir + os.sep + "output.txt", output)


def process_inference_input(input_file):
    with io.open(input_file) as fp:
        texts = [l.strip() for l in fp.readlines()]
    tokenizer = SpacyInstance(disable=["tagger", "parser", "ner"])
    examples = []
    for i, t in enumerate(texts):
        examples.append(TokenClsInputExample(str(i), t, tokenizer.tokenize(t)))
    return examples
