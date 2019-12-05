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
import argparse
from argparse import _SubParsersAction

from nlp_architect.cli.cmd_registry import CMD_REGISTRY
from nlp_architect.version import NLP_ARCHITECT_VERSION


def nlp_train_cli():
    prog_name = 'nlp-train'
    desc = 'NLP Architect Train CLI [{}]'.format(NLP_ARCHITECT_VERSION)
    parser = argparse.ArgumentParser(description=desc, prog=prog_name)
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s v{}'.format(NLP_ARCHITECT_VERSION))
    parser.set_defaults(func=lambda _: parser.print_help())
    subparsers = parser.add_subparsers(title='Models', metavar='')
    for model in CMD_REGISTRY['train']:
        sp = subparsers.add_parser(model['name'],
                                   description=model['description'],
                                   help=model['description'])
        model['arg_adder'](sp)
        sp.set_defaults(func=model['fn'])
    
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

def nlp_inference_cli():
    prog_name = 'nlp-inference'
    desc = 'NLP Architect Inference CLI [{}]'.format(NLP_ARCHITECT_VERSION)
    parser = argparse.ArgumentParser(description=desc, prog=prog_name)
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s v{}'.format(NLP_ARCHITECT_VERSION))
    parser.set_defaults(func=lambda _: parser.print_help())
    subparsers = parser.add_subparsers(title='Models', metavar='')
    for model in CMD_REGISTRY['inference']:
        sp = subparsers.add_parser(model['name'],
                                   description=model['description'],
                                   help=model['description'])
        model['arg_adder'](sp)
        sp.set_defaults(func=model['fn'])
    
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


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
