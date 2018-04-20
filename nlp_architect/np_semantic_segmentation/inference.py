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

from __future__ import unicode_literals, print_function, division, \
    absolute_import
import csv
import os
import io
from neon.backends import gen_backend
from neon.util.argparser import NeonArgparser
from nlp_architect.models.np_semantic_segmentation.model import NpSemanticSegClassifier
from nlp_architect.models.np_semantic_segmentation.data import NpSemanticSegData
import nlp_architect.models.np_semantic_segmentation.data


def classify_collocation(dataset):
    """
    Classify the dataset by the given trained model
    Args:
        dataset: NpSemanticSegData object containing the dataset
    Returns:
        the output of the final layer for the entire Dataset
    """
    # load existing model
    model_path = args.model
    if not os.path.isabs(model_path):
        # handle case using default value\relative paths
        model_path = os.path.join(os.path.dirname(__file__), model_path)
    loaded_model = NpSemanticSegClassifier()
    loaded_model.load(model_path)
    print("Model loaded")
    # arrange the data
    return loaded_model.get_outputs(dataset.train_set)


def print_evaluation(y_test, predictions):
    """
    Print evaluation of the model's predictions comparing to the given y labels (if given)
    Args:
        y_test: list of the labels given in the data
        predictions: the model's predictions
    """
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    for y_true, prediction in zip(y_test, predictions):
        if prediction == 1:
            if y_true == 1:
                tp = tp + 1
            else:
                fp = fp + 1
        elif y_true == 0:
            tn = tn + 1
        else:
            fn = fn + 1
    acc = 100 * ((tp + tn) / len(predictions))
    prec = 100 * (tp / (tp + fp))
    rec = 100 * (tp / (tp + fn))
    print("Model statistics:\naccuracy: {0:.2f}\nprecision: {1:.2f}"
          "\nrecall: {2:.2f}\n".format(acc, prec, rec))


def write_results(predictions):
    """
    Write csv file of predication results to specified --output
    Args:
        predictions:
            the model's predictions
    """
    results_list = predictions.tolist()
    out_file = io.open(args.output, 'w', encoding='utf-8')
    writer = csv.writer(out_file, delimiter=',', quotechar='"')
    for result in results_list:
        writer.writerow([result])
    out_file.close()
    print("Results of inference saved in {0}".format(args.output))


if __name__ == "__main__":
    # parse the command line arguments
    parser = NeonArgparser()
    parser.add_argument('--data', default='datasets/prepared_data.csv',
                        help='prepared data CSV file path')
    parser.add_argument('--model', help='path to the trained model file')
    parser.add_argument('--print_stats', default=False, type=bool,
                        help='print evaluation stats for the model '
                             'predictions - if your data has tagging')
    parser.add_argument('--output', default="datasets/inference_data.csv",
                        help='path to location for inference output file')
    args = parser.parse_args()
    # generate backend
    be = gen_backend(batch_size=10)
    data_set = NpSemanticSegData(args.data, train_to_test_ratio=1)

    results = classify_collocation(data_set)
    if args.print_stats and (data_set.is_y_labels is not None):
        y_labels = data.extract_y_labels(args.data)
        print_evaluation(y_labels, results.argmax(1))
    write_results(results.argmax(1))
