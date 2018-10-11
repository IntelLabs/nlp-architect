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
import errno
import os
from os import path

import pytest
from nlp_architect.pipelines.spacy_np_annotator import NPAnnotator, \
    get_noun_phrases, SpacyNPAnnotator
from nlp_architect.utils.io import download_unlicensed_file
from nlp_architect.utils.text import SpacyInstance, try_to_load_spacy

MODEL_URL = 'http://nervana-modelzoo.s3.amazonaws.com/NLP/chunker/'
MODEL_FILE = 'model.h5'
MODEL_INFO = 'model_info.dat.params'
local_models_path = path.join(path.dirname(path.realpath(__file__)), 'fixtures/data/chunker')

if not try_to_load_spacy('en'):
    pytest.skip("\n\nSkipping test_spacy_np_annotator.py. Reason: 'spacy en' model not installed."
                "Please see https://spacy.io/models/ for installation instructions.\n"
                "The terms and conditions of the data set and/or model license apply.\n"
                "Intel does not grant any rights to the data and/or model files.\n",
                allow_module_level=True)


def check_dir():
    if not os.path.exists(local_models_path):
        try:
            os.makedirs(local_models_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def download(url, filename, local_path):
    if not os.path.exists(local_path):
        download_unlicensed_file(url, filename, local_path)


@pytest.fixture(scope="session", autouse=True)
def setup():
    check_dir()


@pytest.fixture
def model_path():
    path_to_model = path.join(local_models_path, MODEL_FILE)
    download(MODEL_URL, MODEL_FILE, path_to_model)
    return path_to_model


@pytest.fixture
def settings_path():
    path_to_model = path.join(local_models_path, MODEL_INFO)
    download(MODEL_URL, MODEL_INFO, path_to_model)
    return path_to_model


def test_np_annotator_load(model_path, settings_path):
    assert NPAnnotator.load(model_path, settings_path)


@pytest.mark.parametrize('text', ['The quick brown fox jumped over the lazy dog. '
                                  'The quick fox jumped.'])
@pytest.mark.parametrize('phrases', [['lazy dog']])
def test_np_annotator_linked(model_path, settings_path, text, phrases):
    annotator = SpacyInstance(model='en', disable=['textcat', 'ner', 'parser']).parser
    annotator.add_pipe(annotator.create_pipe('sentencizer'), first=True)
    annotator.add_pipe(NPAnnotator.load(model_path, settings_path), last=True)
    doc = annotator(text)
    noun_phrases = [p.text for p in get_noun_phrases(doc)]
    for p in phrases:
        assert p in noun_phrases


@pytest.mark.parametrize('text', ['The quick brown fox jumped over the lazy dog. '
                                  'The quick fox jumped.'])
@pytest.mark.parametrize('phrases', [['lazy dog']])
def test_spacy_np_annotator(model_path, settings_path, text, phrases):
    annotator = SpacyNPAnnotator(model_path, settings_path, spacy_model='en', batch_size=32)
    spans = annotator(text)
    for p in phrases:
        assert p in spans
