# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
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
"""
This script replicates the experiments run in the following paper for music data.
Bai, Shaojie, J. Zico Kolter, and Vladlen Koltun. "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling." arXiv preprint arXiv:1803.01271 (2018).

To compare with the original implementation, run
python ./language_modeling_with_tcn.py --batch_size 16 --dropout 0.45 --epochs 100 --ksize 3
--levels 4 --seq_len 60 --nhid 600 --em_len 600 --em_dropout 0.25 --lr 4 --grad_clip_value 0.35
--results_dir ./ --dataset PTB

python ./language_modeling_with_tcn.py --batch_size 16 --dropout 0.5 --epochs 100 --ksize 3
--levels 5 --seq_len 60 --nhid 1000 --em_len 400 --em_dropout 0.25 --lr 4 --grad_clip_value 0.35
--results_dir ./ --dataset WikiText-103

"""
import argparse
import os

from examples.word_language_model_with_tcn.mle_language_model.lm_model import TCNForLM
from nlp_architect.data.ptb import PTBDataLoader, PTBDictionary
from nlp_architect.utils.io import validate_existing_directory, validate_existing_filepath, \
    validate_parent_exists, check_size


def main(args):
    """
    Main function
    Args:
        args: output of argparse with all input arguments

    Returns:
        None
    """
    hidden_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize
    dropout = args.dropout
    em_dropout = args.em_dropout
    seq_len = args.seq_len
    batch_size = args.batch_size
    n_epochs = args.epochs
    embedding_size = args.em_len
    datadir = os.path.abspath(args.datadir)
    results_dir = os.path.abspath(args.results_dir)

    ptb_dict = PTBDictionary(data_dir=datadir, dataset=args.dataset)
    ptb_dataset_train = PTBDataLoader(ptb_dict, data_dir=datadir, seq_len=seq_len,
                                      split_type="train", skip=seq_len / 2, dataset=args.dataset)
    ptb_dataset_valid = PTBDataLoader(ptb_dict, data_dir=datadir, seq_len=seq_len,
                                      split_type="valid", loop=False, skip=seq_len / 2,
                                      dataset=args.dataset)
    ptb_dataset_test = PTBDataLoader(ptb_dict, data_dir=datadir, seq_len=seq_len,
                                     split_type="test", loop=False, skip=seq_len / 2,
                                     dataset=args.dataset)

    n_train = ptb_dataset_train.n_train
    num_words = len(ptb_dataset_train.idx2word)
    n_per_epoch = int(n_train * 1.0 / batch_size)
    if not args.inference:
        num_iterations = int(n_train * n_epochs * 1.0 / batch_size)
        print("Training examples per epoch: {}, num iterations per epoch: {}"
              .format(n_train, num_iterations // n_epochs))
    else:
        num_iterations = 0

    model = TCNForLM(seq_len, embedding_size, hidden_sizes, kernel_size=kernel_size,
                     dropout=dropout)

    model.build_train_graph(num_words=num_words, max_gradient_norm=args.grad_clip_value,
                            em_dropout=em_dropout)

    if not args.inference:
        model.run(
            {"train": ptb_dataset_train, "valid": ptb_dataset_valid, "test": ptb_dataset_test},
            args.lr, num_iterations=num_iterations, log_interval=n_per_epoch,
            result_dir=results_dir, ckpt=None)
    else:
        sequences = model.run_inference(args.ckpt, num_samples=args.num_samples,
                                        sos=ptb_dict.sos_symbol, eos=ptb_dict.eos_symbol)
        for seq in sequences:
            sentence = []
            for idx in seq:
                while idx == ptb_dict.sos_symbol:
                    continue
                sentence.append(ptb_dict.idx2word[idx])
            print(" ".join(sentence) + "\n")


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--seq_len', type=int, action=check_size(0, 1000),
                    help="Number of time points in each input sequence",
                    default=60)
PARSER.add_argument('--results_dir', type=validate_parent_exists,
                    help="Directory to write results to",
                    default=os.path.expanduser('~/results'))
PARSER.add_argument('--dropout', type=float, default=0.45, action=check_size(0, 1),
                    help='dropout applied to layers, value in [0, 1] (default: 0.45)')
PARSER.add_argument('--ksize', type=int, default=3, action=check_size(0, 10),
                    help='kernel size (default: 3)')
PARSER.add_argument('--levels', type=int, default=4, action=check_size(0, 10),
                    help='# of levels (default: 4)')
PARSER.add_argument('--lr', type=float, default=4, action=check_size(0, 100),
                    help='initial learning rate (default: 4)')
PARSER.add_argument('--nhid', type=int, default=600, action=check_size(0, 1000),
                    help='number of hidden units per layer (default: 600)')
PARSER.add_argument('--em_len', type=int, default=600, action=check_size(0, 10000),
                    help='Length of embedding (default: 600)')
PARSER.add_argument('--em_dropout', type=float, default=0.25, action=check_size(0, 1),
                    help='dropout applied to layers, value in [0, 1] (default: 0.25)')
PARSER.add_argument('--grad_clip_value', type=float, default=0.35, action=check_size(0, 10),
                    help='value to clip each element of gradient')
PARSER.add_argument('--batch_size', type=int, default=16, action=check_size(0, 512),
                    help='Batch size')
PARSER.add_argument('--epochs', type=int, default=100, action=check_size(0, 1000),
                    help='Number of epochs')
PARSER.add_argument('--datadir', type=validate_existing_directory,
                    default=os.path.expanduser('~/data'),
                    help='dir to download data if not already present')
PARSER.add_argument('--dataset', type=str, default="PTB", choices=['PTB', 'WikiText-103'],
                    help='dataset name')
PARSER.add_argument('--inference', action='store_true',
                    help='whether to run in inference mode')
PARSER.add_argument('--ckpt', type=validate_existing_filepath,
                    help='checkpoint file')
PARSER.add_argument('--num_samples', type=int, default=10, action=check_size(0, 10000),
                    help='number of samples to generate during inference')
PARSER.set_defaults()
ARGS = PARSER.parse_args()
main(ARGS)
