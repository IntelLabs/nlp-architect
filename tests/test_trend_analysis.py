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
from os import path

import pytest
import os
from nlp_architect.solutions.trend_analysis import topic_extraction, trend_analysis
from nlp_architect.utils import LIBRARY_STORAGE_PATH
from nlp_architect.utils.text import try_to_load_spacy

if not try_to_load_spacy('en'):
    pytest.skip("\n\nSkipping test_spacy_np_annotator.py. Reason: 'spacy en' model not installed."
                "Please see https://spacy.io/models/ for installation instructions.\n"
                "The terms and conditions of the data set and/or model license apply.\n"
                "Intel does not grant any rights to the data and/or model files.\n",
                allow_module_level=True)

current_dir = os.path.dirname(os.path.realpath(__file__))
ta_path = path.join(LIBRARY_STORAGE_PATH, 'trend-analysis-data')
target_corpus_path = path.join(ta_path, 'target_corpus.csv')
reference_corpus_path = path.join(ta_path, 'reference_corpus.csv')


@pytest.fixture
def input_data_path():
    return os.path.join(current_dir, 'fixtures/data/trend_analysis_test_data')


@pytest.fixture
def output_folder_path():
    return ta_path


@pytest.fixture
def filter_data_path(output_folder_path):
    return os.path.join(output_folder_path, 'filter_phrases.csv')


@pytest.fixture
def graph_data_path(output_folder_path):
    return os.path.join(output_folder_path, 'graph_data.csv')


@pytest.fixture
def model_folder_path(output_folder_path):
    return os.path.join(output_folder_path, 'W2V_Models')


@pytest.fixture
def unified_corpus_path(output_folder_path):
    return os.path.join(output_folder_path, 'corpus.txt')


def test_topic_extraction(input_data_path, output_folder_path):
    tar_corpus_path = os.path.join(input_data_path, 'target_corpus')
    ref_corpus_path = os.path.join(input_data_path, 'reference_corpus')
    topic_extraction.main(tar_corpus_path, ref_corpus_path,
                          single_thread=True, no_train=False, url=False)
    assert os.path.isfile(path.join(ta_path, 'reference_corpus.csv'))
    assert os.path.isfile(path.join(ta_path, 'target_corpus.csv'))
    assert os.path.isfile(os.path.join(output_folder_path,
                                       'W2V_Models/model.bin'))


def test_trend_analysis(filter_data_path, graph_data_path):
    target_corpus_path = path.join(ta_path, 'target_corpus.csv')
    reference_corpus_path = path.join(ta_path, 'reference_corpus.csv')
    trend_analysis.analyze(target_corpus_path, reference_corpus_path,
                           target_corpus_path, reference_corpus_path)
    assert os.path.isfile(filter_data_path)
    assert os.path.isfile(graph_data_path)


# def teardown_module():
#     os.remove(target_corpus_path)
#     os.remove(reference_corpus_path)
#     out_path = output_folder_path()
#     ui_output_files = [f for f in os.listdir(out_path) if
#                        f.endswith(".csv") or f.endswith(".txt")]
#     for f in ui_output_files:
#         os.remove(os.path.join(out_path, f))
#     model_output_files = [f for f in os.listdir(model_folder_path(out_path))]
#     for f in model_output_files:
#         os.remove(os.path.join(model_folder_path(out_path), f))
