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
import argparse
import socket
from os import path
from subprocess import run

SOLUTIONS_PATH = path.dirname(path.realpath(__file__))

solution_uis = {
    'set_expansion': path.join(SOLUTIONS_PATH, 'set_expansion', 'ui'),
    'trend_analysis': path.join(SOLUTIONS_PATH, 'trend_analysis', 'ui')
}


def check_if_ip(ip_str):
    try:
        socket.inet_aton(ip_str)
        return True
    except socket.error:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution', type=str, choices=['set_expansion', 'trend_analysis'],
                        help='Solution UI to initialize')
    parser.add_argument('--address', type=str, default='',
                        help='IP address to use for UI server ')
    parser.add_argument('--port', type=int, default=1010,
                        help='Port number')
    args = parser.parse_args()
    if args.address and not check_if_ip(args.address):
        print('given address is not in a valid ip address format')
        exit(1)

    cmd_str = 'bokeh serve --show {} --port {}'.format(solution_uis[args.solution], args.port)
    if args.address:
        cmd_str += ' --address={} ' \
                   '--allow-websocket-origin={}:{}'.\
            format(args.port, args.address, args.port)

    run(cmd_str)
