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
# pylint: disable=redefined-outer-name

import os
from os import path

from neon.backends import gen_backend

from examples.np_semantic_segmentation.inference import classify_collocation, extract_y_labels, \
    print_evaluation, write_results
from examples.np_semantic_segmentation.train import train_mlp_classifier
from examples.np_semantic_segmentation.data \
    import NpSemanticSegData, read_csv_file_data


def get_data_real_path():
    return path.join(path.dirname(path.realpath(__file__)), 'fixtures/data')


def test_model_training():
    """
    Test model end2end training
    """
    data_path = path.join(get_data_real_path(), 'np_semantic_segmentation_prepared_data.csv')
    model_path = path.join(get_data_real_path(), 'np_semantic_segmentation.prm')
    num_epochs = 200
    gen_backend(batch_size=64, backend='cpu')
    # load data sets from file
    data_set = NpSemanticSegData(data_path, train_to_test_ratio=0.8)
    # train the mlp classifier
    train_mlp_classifier(data_set, model_path, num_epochs, {})
    assert path.isfile(path.join(get_data_real_path(), 'np_semantic_segmentation.prm')) is True


def test_model_inference():
    """
    Test model end2end inference
    """
    data_path = path.join(get_data_real_path(), 'np_semantic_segmentation_prepared_data.csv')
    output_path = path.join(get_data_real_path(), 'np_semantic_segmentation_output.csv')
    model_path = path.join(get_data_real_path(), 'np_semantic_segmentation.prm')
    num_epochs = 200
    callback_args = {}
    gen_backend(batch_size=64, backend='cpu')
    print_stats = False
    data_set = NpSemanticSegData(data_path, train_to_test_ratio=1)
    results = classify_collocation(data_set, model_path, num_epochs, callback_args)
    if print_stats and (data_set.is_y_labels is not None):
        y_labels = extract_y_labels(data_path)
        print_evaluation(y_labels, results.argmax(1))
    write_results(results.argmax(1), output_path)
    assert \
        path.isfile(path.join(get_data_real_path(), 'np_semantic_segmentation_output.csv')) is True
    input_reader_list = read_csv_file_data(data_path)
    output_reader_list = read_csv_file_data(output_path)
    assert len(output_reader_list) == len(input_reader_list) - 1
    os.remove(model_path)
    os.remove(output_path)
