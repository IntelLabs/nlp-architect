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
from importlib import import_module
import logging
import io
import json
import os.path
import gzip
import falcon

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Utility functions, need to move to a util file
def gzip_str(g_str):
    """
    Transform string to GZIP coding

    Args:
        g_str (str): string of data

    Returns:
        GZIP bytes data
    """
    compressed_str = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed_str, mode='w') as file_out:
        file_out.write((json.dumps(g_str).encode()))
    bytes_obj = compressed_str.getvalue()
    return bytes_obj


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
    if (resp_format == "json") or ('json' in resp_format) or (not resp_format):
        # if not specified resp_format then default is json
        return json.dumps(parsed_doc)
    if resp_format == "gzip" or 'gzip' in resp_format:
        return gzip_str(parsed_doc)


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
########################################

class Service(object):
    def __init__(self, service_name):
        self.service_type = None
        self.service = self.load_service(service_name)

    def on_post(self, req, resp):
        """
        Handles POST requests

        Args:
            req (:obj:`falcon.Request`): the client’s HTTP POST request
            resp (:obj:`falcon.Response`):: the server's HTTP response
        """
        logger.info('handle POST request')
        resp.status = falcon.HTTP_200
        set_headers(resp)
        headers = parse_headers(req.headers)
        req_content_type = headers["CONTENT-TYPE"]
        # req_content_encoding = headers["CONTENT-ENCODING"]
        resp_format = headers["RESPONSE-FORMAT"]
        if req_content_type == "application/json" or "application/json" in req_content_type:
            logger.info('Json request')
            try:
                input_json = req.media
                if isinstance(input_json, list):
                    input_json = json.loads(input_json[0])
                input_docs = input_json["docs"]
            except Exception as ex:
                raise falcon.HTTPError(falcon.HTTP_400, 'Error', ex)
        elif req_content_type == "application/gzip":
            logger.info('Gzip request')
            try:
                original_data = gzip.decompress(req.stream.read())
                input_docs = json.loads(str(original_data, 'utf-8'))["docs"]
            except Exception as ex:
                raise falcon.HTTPError(falcon.HTTP_400, 'Error', ex)
        else:
            logger.info('Bad Request Type')
            msg = 'Doc type not allowed. Must be Gzip or Json'
            raise falcon.HTTPBadRequest('Bad request', msg)
        parsed_doc = self.get_service_inference(input_docs, headers)
        logger.info('parsed document processing done')
        resp.body = format_response(resp_format, parsed_doc)

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
        for doc in docs:
            inference_doc = self.service.inference(doc["doc"])
            if self.service_type != "not_spacy":
                parsed_doc = inference_doc.displacy_doc()
                doc_dic = {"id": doc["id"], "doc": parsed_doc}
                if headers['IS-HTML'] is not None and eval(headers['IS-HTML']): # Potentially a security risk
                    # this is a visualizer request - add type of service (core/annotate) to response
                    doc_dic["type"] = self.service_type
                response_data.append(doc_dic)
            else:
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
        with open(os.path.join(package_home(globals()), "services.json")) as prop_file:
            properties = json.load(prop_file)
        folder_path = properties["api_folders_path"]
        model_relative_path = None
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
        print(module_path)
        module_name = extract_module_name(model_relative_path)
        print(str(module_name))
        print(folder_path + module_path)
        module = import_module(folder_path + module_path)
        class_api = getattr(module, module_name)
        upload_service = class_api()
        upload_service.load_model()
        self.service_type = properties[name]["type"]
        return upload_service
