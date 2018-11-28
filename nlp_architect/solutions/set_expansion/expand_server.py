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

import socketserver
import argparse
import pickle
import logging
import sys
from nlp_architect.solutions.set_expansion import set_expand
from nlp_architect.utils.io import validate_existing_filepath, check_size
from nlp_architect.solutions.set_expansion.prepare_data import load_parser, extract_noun_phrases

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    A simple server to load the w2v model and handle expand requests from the
    ui
    """

    def handle(self):
        logger.info("handling expand request")
        res = ''
        self.data = pickle.loads(self.request.recv(10240))
        logger.info('request data: %s', self.data)
        req = self.data[0]
        if req == 'get_vocab':
            logger.info('getting vocabulary')
            res = se.get_vocab()
        elif req == 'in_vocab':
            term = self.data[1]
            res = se.in_vocab(term)
        elif req == 'get_group':
            term = self.data[1]
            res = se.get_group(term)
        elif req == 'annotate':
            seed = self.data[1]
            text = self.data[2]
            res = self.annotate(text, seed)
            logger.info("res:%s", str(res))
        elif req == 'expand':
            logger.info('expanding')
            data = [x.strip() for x in self.data[1].split(',')]
            res = se.expand(data)
        logger.info('compressing response')
        packet = pickle.dumps(res)
        logger.info('response length= %s', str(len(packet)))
        logger.info('sending response')
        self.request.sendall(packet)
        logger.info('done')

    @staticmethod
    def annotate(text, seed):
        np_list = []
        docs = [text]
        spans = extract_noun_phrases(docs, nlp, args.chunker)
        for x in spans:
            np = x.text
            if np not in np_list:
                np_list.append(np)
        logger.info("np_list=%s", str(np_list))
        return se.similarity(np_list, seed, args.similarity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='expand_server.py')
    parser.add_argument('model_path', metavar='model_path',
                        type=validate_existing_filepath,
                        help='a path to the w2v model file')
    parser.add_argument('--host', type=str, default='localhost',
                        help='set port for the server', action=check_size(1, 20))
    parser.add_argument('--port', type=int, default=1234,
                        help='set port for the server', action=check_size(0, 65535))
    parser.add_argument('--grouping', action='store_true', default=False, help='grouping mode')
    parser.add_argument('--similarity', default=0.5, type=float,
                        action=check_size(0, 1), help='similarity threshold')
    parser.add_argument('--chunker', type=str, choices=['spacy', 'nlp_arch'],
                        help='spacy chunker or \'nlp_arch\' for NLP Architect NP Extractor')
    args = parser.parse_args()

    port = args.port
    model_path = args.model_path
    logger.info("loading model")
    se = set_expand.SetExpand(model_path, grouping=args.grouping)
    logger.info("loading chunker")
    nlp = load_parser(args.chunker)
    logger.info("loading server")
    HOST, PORT = args.host, port
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)
    logger.info("server loaded")
    server.serve_forever()
