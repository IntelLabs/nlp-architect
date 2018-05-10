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
# ****************************************************************************
"""
Most Common Word Sense - Train MLP classifier and evaluate it.
"""

import logging
import pickle

import numpy as np
from neon.backends import gen_backend
from neon.data import ArrayIterator
from neon.util.argparser import NeonArgparser

from nlp_architect.models.most_common_word_sense import MostCommonWordSense
from nlp_architect.utils.io import validate_existing_directory, validate_existing_filepath

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def most_common_word_train(x_train, y_train, x_valid, y_valid):
    """
    Train an MLP model, save it and evaluate it

    Args:
        x_train: x data for training
        y_train: y data for training
        x_valid: x data for validation
        y_valid: x data for validation

    Returns:
        str: reslts, predicted values by the model

    """

    # train set
    x_train = np.array(x_train)
    y_train1 = np.array(y_train)
    train_set = ArrayIterator(X=x_train, y=y_train1, make_onehot=False)

    # validation set
    x_valid = np.array(x_valid)
    y_valid1 = np.array(y_valid)
    valid_set = ArrayIterator(X=x_valid, y=y_valid1, make_onehot=False)

    mlp_model = MostCommonWordSense(args.rounding, args.callback_args, args.epochs)
    # build model
    mlp_model.build()
    # train
    mlp_model.fit(valid_set, train_set)
    # save model
    mlp_model.save(args.model_prm)

    # evaluation
    error_rate = mlp_model.eval(valid_set)
    logger.info('Mis-classification error on validation set= %.1f%%' % (error_rate * 100))

    reslts = mlp_model.get_outputs(valid_set)

    return reslts

# -------------------------------------------------------------------------------------#


if __name__ == "__main__":
    # parse the command line arguments
    parser = NeonArgparser()
    parser.add_argument('--data_set_file', default='data/data_set.pkl',
                        type=validate_existing_filepath,
                        help='train and validation sets path')
    parser.add_argument('--model_prm', default='data/mcs_model.prm',
                        type=validate_existing_directory,
                        help=' trained model full path')

    args = parser.parse_args()

    # generate backend, it is optional to change to backend='mkl'
    be = gen_backend(backend='cpu', batch_size=10)

    # read training and validation data file
    with open(args.data_set_file, 'rb') as fp:
        data_in = pickle.load(fp)

    X_train = data_in['X_train']
    X_valid = data_in['X_valid']
    Y_train = data_in['y_train']
    Y_valid = data_in['y_valid']

    logger.info('training set size: ' + repr(len(Y_train)))
    logger.info('validation set size: ' + repr(len(Y_valid)))

    results = most_common_word_train(x_train=X_train, y_train=Y_train, x_valid=X_valid,
                                     y_valid=Y_valid)
