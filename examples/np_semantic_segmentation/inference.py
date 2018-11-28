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
import csv
import os

from .data import NpSemanticSegData, absolute_path

from nlp_architect.models.np_semantic_segmentation import NpSemanticSegClassifier
from nlp_architect.utils.io import validate_existing_filepath, validate_parent_exists


def classify_collocation(test_set, model_file_path, num_epochs, callback_args=None):
    """
    Classify the dataset by the given trained model

    Args:
        model_file_path (str): model path
        num_epochs (int): number of epochs
        callback_args (dict): callback_arg
        test_set (:obj:`core_models.np_semantic_segmentation.data.NpSemanticSegData`):
            NpSemanticSegData object containing the dataset

    Returns:
        the output of the final layer for the entire Dataset
    """
    # load existing model
    if not os.path.isabs(model_file_path):
        # handle case using default value\relative paths
        model_file_path = os.path.join(os.path.dirname(__file__), model_file_path)
    loaded_model = NpSemanticSegClassifier(num_epochs, callback_args)
    loaded_model.load(model_file_path)
    print("Model loaded")
    # arrange the data
    return loaded_model.get_outputs(test_set['X'])


def print_evaluation(y_test, predictions):
    """
    Print evaluation of the model's predictions comparing to the given y labels (if given)

    Args:
        y_test (list(str)): list of the labels given in the data
        predictions(obj:`numpy.ndarray`): the model's predictions
    """
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    for y_true, prediction in zip(y_test, [round(p[0]) for p in predictions.tolist()]):
        if prediction == 1:
            if y_true == 1:
                tp = tp + 1
            else:
                fp = fp + 1
        elif y_true == 0:
            tn = tn + 1
        else:
            fn = fn + 1
    if tp + fn == 0:
        fn = 1
    if tp == 0:
        tp = 1
    acc = 100 * ((tp + tn) / len(predictions))
    prec = 100 * (tp / (tp + fp))
    rec = 100 * (tp / (tp + fn))
    print("Model statistics:\naccuracy: {0:.2f}\nprecision: {1:.2f}"
          "\nrecall: {2:.2f}\n".format(acc, prec, rec))


def write_results(predictions, output):
    """
    Write csv file of predication results to specified --output

    Args:
        output (str): output file path
        predictions:
            the model's predictions
    """
    results_list = [round(p[0]) for p in predictions.tolist()]
    with open(output, 'w', encoding='utf-8') as out_file:
        writer = csv.writer(out_file, delimiter=',', quotechar='"')
        for result in results_list:
            writer.writerow([result])
    print("Results of inference saved in {0}".format(output))


if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.set_defaults(epochs=200)
    parser.add_argument('--data', help='prepared data CSV file path',
                        type=validate_existing_filepath)
    parser.add_argument('--model', help='path to the trained model file',
                        type=validate_existing_filepath)
    parser.add_argument('--print_stats', action='store_true', default=False,
                        help='print evaluation stats for the model predictions - if '
                             'your data has tagging')
    parser.add_argument('--output', help='path to location for inference output file',
                        type=validate_parent_exists)
    args = parser.parse_args()
    data_path = absolute_path(args.data)
    model_path = absolute_path(args.model)
    print_stats = args.print_stats
    output_path = absolute_path(args.output)
    data_set = NpSemanticSegData(data_path)
    results = classify_collocation(data_set.test_set, model_path, args.epochs)
    if print_stats and (data_set.is_y_labels is not None):
        y_labels = data_set.test_set_y
        print_evaluation(y_labels, results)
    write_results(results, output_path)
