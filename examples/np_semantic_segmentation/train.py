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

import argparse

from .data import NpSemanticSegData, absolute_path

from nlp_architect.models.np_semantic_segmentation import NpSemanticSegClassifier
from nlp_architect.utils.io import validate_existing_filepath, validate_parent_exists, validate


def train_mlp_classifier(dataset, model_file_path, epochs, callback_args=None):
    """
    Train the np_semantic_segmentation mlp classifier
    Args:
        model_file_path (str): model path
        epochs (int): number of epochs
        callback_args (dict): callback_arg
        dataset: NpSemanticSegData object containing the dataset

    Returns:
        print error_rate, test_accuracy_rate and precision_recall_rate evaluation from the model

    """
    model = NpSemanticSegClassifier(epochs, callback_args)
    input_dim = dataset.train_set_x.shape[1]
    model.build(input_dim)
    # run fit
    model.fit(dataset.train_set)
    # save model params
    model.save(model_file_path)
    # set evaluation error rates
    loss, binary_accuracy, precision, recall, f1 = model.eval(dataset.test_set)
    print('loss = %.1f%%' % (loss))
    print('Test binary_accuracy rate = %.1f%%' % (binary_accuracy * 100))
    print('Test precision rate = %.1f%%' % (precision * 100))
    print('Test recall rate = %.1f%%' % (recall * 100))
    print('Test f1 rate = %.1f%%' % (f1 * 100))


if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.set_defaults(epochs=200)
    parser.add_argument('--data', type=validate_existing_filepath,
                        help='Path to the CSV file where the prepared dataset is saved')
    parser.add_argument('--model_path', type=validate_parent_exists,
                        help='Path to save the model')
    args = parser.parse_args()
    validate((args.epochs, int, 1, 100000))
    data_path = absolute_path(args.data)
    model_path = absolute_path(args.model_path)
    num_epochs = args.epochs
    # load data sets from file
    data_set = NpSemanticSegData(data_path, train_to_test_ratio=0.8)
    # train the mlp classifier
    train_mlp_classifier(data_set, model_path, args.epochs)
