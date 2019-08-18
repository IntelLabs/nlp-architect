#! /usr/bin/env python
# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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
from subprocess import run

from nlp_architect import LIBRARY_PATH


def run_cmd(command):
    return run(command.split(), shell=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=8000, help="server port")
    args = parser.parse_args()
    port = args.p
    serve_file = LIBRARY_PATH / 'server' / 'serve.py'
    cmd_str = 'hug -p {} -f {}'.format(port, serve_file)
    print('Starting NLP Architect demo server')
    run_cmd(cmd_str)
