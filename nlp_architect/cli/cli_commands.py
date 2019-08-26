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
# CLI definition
from argparse import _SubParsersAction

from nlp_architect.cli.cmd_registry import CMD_REGISTRY


def generic_cmd(cmd_name: str, subtitle: str, description: str, subparsers: _SubParsersAction):
    parser = subparsers.add_parser(cmd_name,
                                   description=description,
                                   help=description)

    subsubparsers = parser.add_subparsers(title=subtitle,
                                          metavar='')
    for model in CMD_REGISTRY[cmd_name]:
        sp = subsubparsers.add_parser(model['name'],
                                      description=model['description'],
                                      help=model['description'])
        model['arg_adder'](sp)
        sp.set_defaults(func=model['fn'])
    parser.set_defaults(func=lambda _: parser.print_help())


def cli_train_cmd(subparsers: _SubParsersAction):
    generic_cmd('train',
                'Available models',
                'Train a model from the library',
                subparsers)


def cli_run_cmd(subparsers: _SubParsersAction):
    generic_cmd('run',
                'Available models',
                'Run a model from the library',
                subparsers)


def cli_process_cmd(subparsers: _SubParsersAction):
    generic_cmd('process',
                'Available data processors',
                'Run a data processor from the library',
                subparsers)


def cli_solution_cmd(subparsers: _SubParsersAction):
    generic_cmd('solution',
                'Available solutions',
                'Run a solution process from the library',
                subparsers)


def cli_serve_cmd(subparsers: _SubParsersAction):
    generic_cmd('serve',
                'Available models',
                'Server a trained model using REST service',
                subparsers)
