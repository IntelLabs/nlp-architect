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
from solutions.set_expansion import set_expand


class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        print("handling expand request")
        res = ''
        self.data = str(self.request.recv(10240).strip(), 'utf-8')
        print('request data: ' + self.data)
        if self.data == 'get_vocab':
            print('getting vocabulary')
            res = se.get_vocab()
        else:
            data = [x.strip() for x in self.data.split(',')]
            print('expanding')
            res = se.expand(data)
        print('compressing response')
        packet = pickle.dumps(res)
        print('response length= ' + str(len(packet)))
        # length = struct.pack('!I', len(packet))
        # packet = length + packet
        print('sending response')
        self.request.sendall(packet)
        print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='expand_server.py')
    parser.add_argument('model_path', metavar='model_path', type=str,
                        help='a path to the w2v model file')
    parser.add_argument('--host',type=str, default='localhost',
                        help='set port for the server')
    parser.add_argument('--port', type=int, default=1234,
                        help='set port for the server')
    args = parser.parse_args()

    port = args.port
    model_path = args.model_path
    print("loading model")
    se = set_expand.SetExpand(model_path)
    print("loading server")
    HOST, PORT = args.host, port
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)
    print("server loaded")
    server.serve_forever()

