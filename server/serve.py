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
import os
import hug
from falcon import status_codes
from service import Service
services = {}

@hug.post()
def inference(request, body, response):
    """Makes an inference to a certain model"""
    # Consider putting model_name as a param
    if(request.headers.get('CONTENT-TYPE') != 'application/json'):
        response.status = status_codes.HTTP_400
        return { 'status': 'Content-Type header must be application/json'}
    model_name = body.get('model_name')
    if(not model_name):
        response.status = status_codes.HTTP_400
        return {'status': 'model_name is required'}
    # If we've already initialized it, no use in reinitializing
    if not services.get(model_name):
        services[model_name] = Service(model_name)
    input_docs = body.get('docs')
    if not isinstance(input_docs, list): # check if it's an array instead
        response.status = status_codes.HTTP_400
        return { 'status': 'request not in proper format '}
    parsed_doc = services[model_name].get_service_inference(input_docs, request.headers)
    return parsed_doc

@hug.static('/')
def static():
    return [os.path.realpath(os.path.join('./', 'server/web_service/static'))]
