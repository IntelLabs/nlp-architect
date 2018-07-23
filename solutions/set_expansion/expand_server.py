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
from solutions.set_expansion import set_expand
from nlp_architect.utils.io import validate_existing_filepath, check_size

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
        self.data = str(self.request.recv(10240).strip(), 'utf-8')
        logger.info('request data: ' + self.data)
        if self.data == 'get_vocab':
            logger.info('getting vocabulary')
            res = se.get_vocab()
        elif 'in_vocab' in self.data:
            term = self.data.split(',')[1]
            res = se.in_vocab(term)
        elif 'get_group' in self.data:
            term = self.data.split(',')[1]
            res = se.get_group(term)
        else:
            data = [x.strip() for x in self.data.split(',')]
            logger.info('expanding')
            res = se.expand(data)
        # logger.info('res: ' + str(res))
        logger.info('compressing response')
        packet = pickle.dumps(res)
        logger.info('response length= ' + str(len(packet)))
        # length = struct.pack('!I', len(packet))
        # packet = length + packet
        logger.info('sending response')
        self.request.sendall(packet)
        logger.info('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='expand_server.py')
    parser.add_argument('model_path', metavar='model_path', type=validate_existing_filepath,
                        help='a path to the w2v model file')
    parser.add_argument('--host', type=str, default='localhost',
                        help='set port for the server', action=check_size(1, 20))
    parser.add_argument('--port', type=int, default=1234,
                        help='set port for the server', action=check_size(0, 65535))
    parser.add_argument('--grouping',action='store_true',default=False,help='grouping mode')
    args = parser.parse_args()

    port = args.port
    model_path = args.model_path
    logger.info("loading model")
    se = set_expand.SetExpand(model_path, grouping=args.grouping)
    logger.info("loading server")
    HOST, PORT = args.host, port
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)
    logger.info("server loaded")
    server.serve_forever()
