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
# pylint: disable=c-extension-no-member
""" REST Server to respond to different API requests """
import gzip
import json
from os import path

import hug
from falcon import status_codes

from nlp_architect.server.service import Service, parse_headers, format_response

services = {}

api = hug.API(__name__)
api.http.add_middleware(hug.middleware.CORSMiddleware(api, max_age=10))


def prefetch_models():
    models = ['machine_comprehension', 'bist', 'ner', 'intent_extraction']
    for model in models:
        services[model] = Service(model)


@hug.get('/comprehension_paragraphs')
def get_paragraphs():
    if not services['machine_comprehension']:
        services['machine_comprehension'] = Service('machine_comprehension')
    return services['machine_comprehension'].get_paragraphs()


# pylint: disable=inconsistent-return-statements
@hug.post()
def inference(request, body, response):
    """Makes an inference to a certain model"""
    print(body)
    if request.headers.get('CONTENT-TYPE') == 'application/gzip':
        try:
            original_data = gzip.decompress(request.stream.read())
            input_docs = json.loads(str(original_data, 'utf-8'))["docs"]
            model_name = json.loads(str(original_data, 'utf-8'))["model_name"]
        except Exception:
            response.status = hug.HTTP_500
            return {'status': 'unexpected gzip error'}
    elif request.headers.get('CONTENT-TYPE') == 'application/json':
        if isinstance(body, str):
            body = json.loads(body)
        model_name = body.get('model_name')
        input_docs = body.get('docs')
    else:
        response.status = status_codes.HTTP_400
        return {'status': 'Content-Type header must be application/json or application/gzip'}
    if not model_name:
        response.status = status_codes.HTTP_400
        return {'status': 'model_name is required'}
    # If we've already initialized it, no use in reinitializing
    if not services.get(model_name):
        services[model_name] = Service(model_name)
    if not isinstance(input_docs, list):  # check if it's an array instead
        response.status = status_codes.HTTP_400
        return {'status': 'request not in proper format '}
    headers = parse_headers(request.headers)
    parsed_doc = services[model_name].get_service_inference(input_docs, headers)
    resp_format = request.headers["RESPONSE-FORMAT"]
    ret = format_response(resp_format, parsed_doc)
    if request.headers.get('CONTENT-TYPE') == 'application/gzip':
        response.content_type = resp_format
        response.body = ret
        # no return due to the fact that hug seems to assume json type upon return
    else:
        return ret


@hug.static('/')
def static():
    """Statically serves a directory to client"""
    return [path.join(path.dirname(path.realpath(__file__)), 'angular-ui/dist/angular-ui')]
    # return [os.path.realpath(os.path.join('./', 'server/angular-ui/dist/angular-ui'))]


@hug.get(['/home', '/visual/{page}', '/annotate/{page}', '/machine_comprehension'],
         output=hug.output_format.file)
def get_index():
    index = path.join(path.dirname(path.realpath(__file__)),
                      'angular-ui/dist/angular-ui/index.html')
    return index


prefetch_models()
