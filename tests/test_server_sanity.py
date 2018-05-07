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

import gzip
import io
import json
import os
from os.path import dirname
from io import open
import sys

import falcon
import pytest
from falcon import testing

# from nlp_architect_server.serve import app
from falcon_multipart.middleware import MultipartMiddleware

import server.serve

sys.path.insert(0, (dirname(dirname(os.path.abspath(__file__)))))

headers = {"clean": "True", "display_post_preprocces": "True",
           "display_tokens": "", "display_token_text": "True"}
server_data_rel_path = 'fixtures/data/server_data/'


def load_test_data(service_name):
    """
    load test data (input and expected response) for given service from 'tests_data.json'

    Args:
        service_name (str):  the service name

    Returns:
        str: the test data of the service
    """
    f = open(os.path.join(os.path.dirname(__file__),server_data_rel_path + 'tests_data.json'), 'r')
    service_test_data = json.loads(f.read())[service_name]
    return service_test_data


def init_client(service_name):
    """
    init dummy client for testing.
    Args:
        service_name(str):  the service name
    Returns:
        client for testing
    """
    app = falcon.API(middleware=[MultipartMiddleware()])
    server.serve.set_server_properties(app, service_name)
    return testing.TestClient(app)


@pytest.mark.parametrize('service_name', ['bist', 'spacy_ner'])
def test_request(service_name):
    client = init_client(service_name)
    test_data = load_test_data(service_name)
    doc = json.dumps(test_data["input"])
    expected_result = json.dumps(test_data["response"])
    headers["Content-Type"] = "application/json"
    headers["format"] = "json"
    response = client.simulate_post('/'+ service_name, body=doc, headers=headers)
    result_doc = json.loads(response.content, encoding='utf-8')
    assert result_doc == json.loads(expected_result)
    assert response.status == falcon.HTTP_OK


@pytest.mark.parametrize('service_name', ['bist', 'spacy_ner'])
def test_gzip_file_request(service_name):
    client = init_client(service_name)
    file_path = os.path.join(os.path.dirname(__file__), server_data_rel_path+ service_name + "_sentences_examples.json.gz")
    file_data = open(file_path, 'rb').read()
    doc = file_data
    expected_result = json.dumps(load_test_data(service_name)["response"])
    headers["Content-Type"] = "application/gzip"
    headers["Content-Encoding"] = "gzip"
    headers["format"] = "gzip"
    response = client.simulate_post('/'+ service_name, body=doc, headers=headers)
    result_doc = get_decompressed_gzip(response.content)
    assert result_doc == json.loads(expected_result)
    assert response.status == falcon.HTTP_OK


@pytest.mark.parametrize('service_name', ['bist', 'spacy_ner'])
def test_json_file_request(service_name):
    client = init_client(service_name)
    file_path = os.path.join(os.path.dirname(__file__), server_data_rel_path + service_name + "_sentences_examples.json")
    doc = open(file_path, 'rb').read()
    expected_result = json.dumps(load_test_data(service_name)["response"])
    headers["Content-Type"] = "application/json"
    headers["format"] = "json"
    response = client.simulate_post('/'+ service_name, body=doc, headers=headers)
    result_doc = json.loads(response.content, encoding='utf-8')
    assert result_doc == json.loads(expected_result)
    assert response.status == falcon.HTTP_OK


def get_decompressed_gzip(req_resp):
    tmp_file = io.BytesIO()
    tmp_file.write(req_resp)
    tmp_file.seek(0)
    with gzip.GzipFile(fileobj=tmp_file, mode='rb') as file_out:
        gunzipped_bytes_obj = file_out.read()
    return json.loads(gunzipped_bytes_obj.decode())
