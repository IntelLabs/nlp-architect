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
import logging

# register all procedures by importing
import nlp_architect.procedures  # noqa: F401
from nlp_architect.cli.cli_commands import cli_train_cmd, cli_run_cmd
from nlp_architect.version import NLP_ARCHITECT_VERSION

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


from nlp_architect.cli.cli_commands import nlp_train_cli, nlp_inference_cli

def run_cli():
    """ Run nlp_architect command line application
    """
    prog_name = 'nlp_architect'
    desc = 'NLP Architect CLI [{}]'.format(NLP_ARCHITECT_VERSION)
    parser = argparse.ArgumentParser(description=desc, prog=prog_name)
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s v{}'.format(NLP_ARCHITECT_VERSION))

    parser.set_defaults(func=lambda _: parser.print_help())
    subparsers = parser.add_subparsers(title='commands', metavar='')
    for command in sub_commands:
        command(subparsers)
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


# sub commands list
sub_commands = [
    cli_train_cmd,
    cli_run_cmd,
]

if __name__ == "__main__":
    run_cli()
