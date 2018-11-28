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
# pylint: disable=c-extension-no-member
import gzip
import io
import json
import os
import sys
from io import open
from os.path import dirname

import hug
import pytest

import nlp_architect.server.serve
from nlp_architect.server.serve import api
from nlp_architect.utils.text import try_to_load_spacy

if not try_to_load_spacy('en'):
    pytest.skip("\n\nSkipping test_server_sanity.py. Reason: 'spacy en' model not installed. "
                "Please see https://spacy.io/models/ for installation instructions.\n"
                "The terms and conditions of the data set and/or model license apply.\n"
                "Intel does not grant any rights to the data and/or model files.\n",
                allow_module_level=True)

sys.path.insert(0, (dirname(dirname(os.path.abspath(__file__)))))

headers = {"clean": "True", "display_post_preprocces": "True",
           "display_tokens": "", "display_token_text": "True",
           "IS-HTML": "False"}
server_data_rel_path = 'fixtures/data/server/'


def load_test_data(service_name):
    """
    load test data (input and expected response) for given service from 'tests_data.json'
    Args:
        service_name (str):  the service name
    Returns:
        str: the test data of the service
    """
    with open(os.path.join(os.path.dirname(__file__), server_data_rel_path + 'tests_data.json'),
              'r') as f:
        service_test_data = json.loads(f.read())[service_name]
    return service_test_data


def assert_response_struct(result_doc, expected_result):
    # 1. assert docs list
    assert isinstance(result_doc, list)
    assert len(result_doc) == len(expected_result)
    # 2. assert the structure of doc item
    result_item = result_doc[0]
    expected_result_item = expected_result[0]
    assert isinstance(result_item, dict)
    for key in expected_result_item.keys():
        assert key in result_item
    # 3. assert the structure of doc item dict
    result_dict = result_item['doc']
    expected_result_dict = expected_result_item['doc']
    if isinstance(result_dict, list):
        result_dict = result_dict[0]
        expected_result_dict = expected_result_dict[0]
    assert isinstance(result_dict, dict)
    for key in expected_result_dict.keys():
        assert key in result_dict
    # 4. check CoreNLPDoc
    if 'sentences' in expected_result_dict.keys():
        assert isinstance(result_dict['sentences'], list)
        # assert sentence:
        assert isinstance(result_dict['sentences'][0], list)
        # assert word-token
        result_word_dict = result_dict['sentences'][0][0]
        expected_result_word_dict = expected_result_item['sentences'][0][0]
        for key in expected_result_word_dict.keys():
            assert key in result_word_dict
    # 5. check HighLevelDoc
    elif 'annotation_set' in expected_result_dict.keys():
        assert isinstance(result_dict['annotation_set'], list)
        assert isinstance(result_dict['spans'], list)
        result_spans = result_dict['spans'][0]
        expected_result_spans = expected_result_dict['spans'][0]
        assert isinstance(result_spans, dict)
        for key in expected_result_spans.keys():
            assert key in result_spans
    # 6. check displacy html rendering input
    elif 'arcs' in expected_result_dict.keys():
        assert isinstance(result_dict['arcs'], list)
        assert isinstance(result_dict['words'], list)
        result_arcs = result_dict['arcs'][0]
        expected_result_arcs = expected_result_dict['arcs'][0]
        assert isinstance(result_arcs, dict)
        for key in expected_result_arcs.keys():
            assert key in result_arcs
        result_words = result_dict['words'][0]
        expected_result_words = expected_result_dict['words'][0]
        assert isinstance(result_words, dict)
        for key in expected_result_words.keys():
            assert key in result_words


@pytest.mark.parametrize('service_name', ['bist', 'ner'])
def test_request(service_name):
    test_data = load_test_data(service_name)
    test_data['input']['model_name'] = service_name
    doc = json.dumps(test_data["input"])
    expected_result = json.dumps(test_data["response"])
    myHeaders = headers.copy()
    myHeaders["content-type"] = "application/json"
    myHeaders["Response-Format"] = "json"
    response = hug.test.post(api, '/inference', body=doc, headers=myHeaders)

    assert_response_struct(response.data, json.loads(expected_result))
    assert response.status == hug.HTTP_OK


@pytest.mark.parametrize('service_name', ['bist', 'ner'])
def test_gzip_file_request(service_name):
    file_path = os.path.join(os.path.dirname(__file__), server_data_rel_path + service_name
                             + "_sentences_examples.json.gz")
    with open(file_path, 'rb') as file_data:
        doc = file_data.read()
    expected_result = json.dumps(load_test_data(service_name)["response"])
    myHeaders = headers.copy()
    myHeaders["content-type"] = "application/gzip"
    myHeaders["Response-Format"] = "gzip"
    myHeaders["content-encoding"] = "gzip"
    response = hug.test.post(api, '/inference', body=doc, headers=myHeaders)
    result_doc = get_decompressed_gzip(response.data)
    assert_response_struct(result_doc, json.loads(expected_result))
    assert response.status == hug.HTTP_OK


@pytest.mark.parametrize('service_name', ['bist', 'ner'])
def test_json_file_request(service_name):
    file_path = os.path.join(os.path.dirname(__file__), server_data_rel_path + service_name
                             + "_sentences_examples.json")
    with open(file_path, 'rb') as file:
        doc = file.read()
    expected_result = json.dumps(load_test_data(service_name)["response"])
    myHeaders = headers.copy()
    myHeaders["Content-Type"] = "application/json"
    myHeaders["RESPONSE-FORMAT"] = "json"
    response = hug.test.post(nlp_architect.server.serve, '/inference', body=doc, headers=myHeaders)
    assert_response_struct(response.data, json.loads(expected_result))
    assert response.status == hug.HTTP_OK


def get_decompressed_gzip(req_resp):
    tmp_file = io.BytesIO()
    tmp_file.write(req_resp)
    tmp_file.seek(0)
    with gzip.GzipFile(fileobj=tmp_file, mode='rb') as file_out:
        gunzipped_bytes_obj = file_out.read()
    return json.loads(gunzipped_bytes_obj.decode())
