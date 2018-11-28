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
""" Service file used to import different features for server """
import json
import logging
import os.path
from importlib import import_module

from nlp_architect.utils.io import gzip_str

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def format_response(resp_format, parsed_doc):
    """
    Transform string of server's response to the requested format

    Args:
        resp_format(str): the desired response format
        parsed_doc: the server's response

    Returns:
        formatted response
    """
    logger.info('preparing response JSON')
    ret = None
    if (resp_format == "json") or ('json' in resp_format) or (not resp_format):
        # if not specified resp_format then default is json
        ret = parsed_doc
    if resp_format == "gzip" or 'gzip' in resp_format:
        ret = gzip_str(parsed_doc)
    return ret


def parse_headers(req_headers):
    """
    Load headers from request to dictionary

    Args:
        req_headers (dict): the request's headers

    Returns:
        dict: dictionary hosting the request headers
    """
    headers_lst = ["CONTENT-TYPE", "CONTENT-ENCODING", "RESPONSE-FORMAT",
                   "CLEAN", "DISPLAY-POST-PREPROCCES",
                   "DISPLAY-TOKENS", "DISPLAY-TOKEN-TEXT", "IS-HTML"]
    headers = {}
    for header_tag in headers_lst:
        if header_tag in req_headers:
            headers[header_tag] = req_headers[header_tag]
        else:
            headers[header_tag] = None
    return headers


def set_headers(res):
    """
    set expected headers for request (CORS)

    Args:
        res (:obj:`falcon.Response`): the request
    """
    res.set_header('Access-Control-Allow-Origin', '*')
    res.set_header("Access-Control-Allow-Credentials", "true")
    res.set_header('Access-Control-Allow-Methods', "GET,HEAD,OPTIONS,POST,PUT")
    res.set_header('Access-Control-Allow-Headers',
                   "Access-Control-Allow-Headers, Access-Control-Allow-Origin,"
                   " Origin,Accept, X-Requested-With, Content-Type, "
                   "Access-Control-Request-Method, "
                   "Access-Control-Request-Headers, Response-Format, clean, "
                   "display-post-preprocces, display-tokens, "
                   "display-token-text")


def package_home(gdict):
    """
    help function for running paths from out-of-class scripts
    """
    filename = gdict["__file__"]
    return os.path.dirname(filename)


def extract_module_name(model_path):
    """
    Extract the module's name from path

    Args:
        model_path(str): the module's class path

    Returns:
        str: the modules name
    """
    class_name = "".join(model_path.split(".")[0].title().split("_"))
    return class_name


class Service(object):
    """Handles loading and inference using specific models"""
    def __init__(self, service_name):
        self.service_type = None
        self.is_spacy = False
        self.service = self.load_service(service_name)

    def get_paragraphs(self):
        return self.service.get_paragraphs()

    # pylint: disable=eval-used
    def get_service_inference(self, docs, headers):
        """
        get parser response from service API

        Args:
            headers (list(str)): the headers of the request
            docs: input received from the request

        Returns:
            the service API output
        """
        logger.info('sending documents to parser')
        response_data = []
        for i, doc in enumerate(docs):
            inference_doc = self.service.inference(doc["doc"])
            if self.is_spacy is True:
                parsed_doc = inference_doc.displacy_doc()
                doc_dic = {"id": doc["id"], "doc": parsed_doc}
                # Potentially a security risk
                if headers['IS-HTML'] is not None and eval(headers['IS-HTML']):
                    # a visualizer requestadd type of service (core/annotate) to response
                    doc_dic["type"] = self.service_type
                response_data.append(doc_dic)
            else:
                inference_doc['id'] = i + 1
                response_data.append(inference_doc)
        return response_data

    def load_service(self, name):
        """
        Initialize and load service from input given name, using "services.json" properties file

        Args:
            name (str):
                The name of service to upload using server

        Returns:
            The loaded service
        """
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "services.json")) \
                as prop_file:
            properties = json.load(prop_file)
        folder_path = properties["api_folders_path"]
        service_name_error = "'{0}' is not an existing service - " \
                             "please try using another service.".format(name)
        if name in properties:
            model_relative_path = properties[name]["file_name"]
        else:
            logger.error(service_name_error)
            raise Exception(service_name_error)
        if not model_relative_path:
            logger.error(service_name_error)
            raise Exception(service_name_error)
        module_path = ".".join(model_relative_path.split(".")[:-1])
        module_name = extract_module_name(model_relative_path)
        module = import_module(folder_path + module_path)
        class_api = getattr(module, module_name)
        upload_service = class_api()
        upload_service.load_model()
        self.service_type = properties[name]["type"]
        self.is_spacy = properties[name].get('spacy', False)
        return upload_service
