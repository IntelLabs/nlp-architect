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
import argparse
import logging
import pickle

import numpy as np

from nlp_architect.models.most_common_word_sense import MostCommonWordSense
from nlp_architect.utils.io import validate_existing_filepath, \
    validate_parent_exists, check_size

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
    train_set = {'X': x_train, 'y': y_train1}

    # validation set
    x_valid = np.array(x_valid)
    y_valid1 = np.array(y_valid)
    valid_set = {'X': x_valid, 'y': y_valid1}

    input_dim = train_set['X'].shape[1]
    mlp_model = MostCommonWordSense(args.epochs, args.batch_size, None)
    # build model
    mlp_model.build(input_dim)
    # train
    mlp_model.fit(train_set)
    # save model
    mlp_model.save(args.model)

    # evaluation
    error_rate = mlp_model.eval(valid_set)
    logger.info('Mis-classification error on validation set= %0.1f', error_rate * 100)

    reslts = mlp_model.get_outputs(valid_set['X'])

    return reslts


if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set_file', default='data/data_set.pkl',
                        type=validate_existing_filepath,
                        help='train and validation sets path')
    parser.add_argument('--model', default='data/mcs_model.h5',
                        type=validate_parent_exists,
                        help='trained model full path')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs',
                        action=check_size(0, 200))
    parser.add_argument('--batch_size', default=50, type=int, help='batch_size',
                        action=check_size(0, 256))

    args = parser.parse_args()

    # read training and validation data file
    with open(args.data_set_file, 'rb') as fp:
        data_in = pickle.load(fp)

    X_train = data_in['X_train']
    X_valid = data_in['X_valid']
    Y_train = data_in['y_train']
    Y_valid = data_in['y_valid']

    logger.info('training set size: %s', str(len(Y_train)))
    logger.info('validation set size: %s', str(len(Y_valid)))

    results = most_common_word_train(x_train=X_train, y_train=Y_train, x_valid=X_valid,
                                     y_valid=Y_valid)
