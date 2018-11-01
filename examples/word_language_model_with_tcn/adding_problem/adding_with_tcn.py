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
This script replicates the experiments in the following paper for the synthetic "adding" data:
Bai, Shaojie, J. Zico Kolter, and Vladlen Koltun. "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling." arXiv preprint arXiv:1803.01271 (2018).

To compare with the original implementation, run
python ./adding_with_tcn.py --batch_size 32 --dropout 0.0 --epochs 20 --ksize 6 --levels 7
--seq_len 200 --log_interval 100 --nhid 27 --lr 0.002 --results_dir ./

python ./adding_with_tcn.py --batch_size 32 --dropout 0.0 --epochs 20 --ksize 7 --levels 7
--seq_len 400 --log_interval 100 --nhid 27 --lr 0.002 --results_dir ./

python ./adding_with_tcn.py --batch_size 32 --dropout 0.0 --epochs 20 --ksize 8 --levels 8
--seq_len 600 --log_interval 100 --nhid 24 --lr 0.002 --results_dir ./
"""
import argparse
import os

from examples.word_language_model_with_tcn.adding_problem.adding_model import TCNForAdding
from examples.word_language_model_with_tcn.toy_data.adding import Adding
from nlp_architect.utils.io import validate_parent_exists, check_size


def main(args):
    """
    Main function
    Args:
        args: output of argparse with all input arguments

    Returns:
        None
    """
    n_features = 2
    hidden_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize
    dropout = args.dropout
    seq_len = args.seq_len
    n_train = 50000
    n_val = 1000
    batch_size = args.batch_size
    n_epochs = args.epochs
    num_iterations = int(n_train * n_epochs * 1.0 / batch_size)
    results_dir = os.path.abspath(args.results_dir)

    adding_dataset = Adding(seq_len=seq_len, n_train=n_train, n_test=n_val)

    model = TCNForAdding(seq_len, n_features, hidden_sizes, kernel_size=kernel_size,
                         dropout=dropout)

    model.build_train_graph(args.lr, max_gradient_norm=args.grad_clip_value)

    model.run(adding_dataset, num_iterations=num_iterations, log_interval=args.log_interval,
              result_dir=results_dir)


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--seq_len', type=int, action=check_size(0, 1000),
                    help="Number of time points in each input sequence",
                    default=200)
PARSER.add_argument('--log_interval', type=int, default=100, action=check_size(0, 10000),
                    help="frequency, in number of iterations, after which loss is evaluated")
PARSER.add_argument('--results_dir', type=validate_parent_exists,
                    help="Directory to write results to", default=os.path.expanduser('~/results'))
PARSER.add_argument('--dropout', type=float, default=0.0, action=check_size(0, 1),
                    help='dropout applied to layers, between 0 and 1 (default: 0.0)')
PARSER.add_argument('--ksize', type=int, default=6, action=check_size(0, 10),
                    help='kernel size (default: 6)')
PARSER.add_argument('--levels', type=int, default=7, action=check_size(0, 10),
                    help='# of levels (default: 7)')
PARSER.add_argument('--lr', type=float, default=2e-3, action=check_size(0, 1),
                    help='initial learning rate (default: 2e-3)')
PARSER.add_argument('--nhid', type=int, default=27, action=check_size(0, 1000),
                    help='number of hidden units per layer (default: 27)')
PARSER.add_argument('--grad_clip_value', type=float, default=10, action=check_size(0, 10),
                    help='value to clip each element of gradient')
PARSER.add_argument('--batch_size', type=int, default=32, action=check_size(0, 512),
                    help='Batch size')
PARSER.add_argument('--epochs', type=int, default=20, action=check_size(0, 1000),
                    help='Number of epochs')
PARSER.set_defaults()
ARGS = PARSER.parse_args()
main(ARGS)
