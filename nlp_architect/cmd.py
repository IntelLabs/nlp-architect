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
import os
from http.server import HTTPServer as BaseHTTPServer, SimpleHTTPRequestHandler
from os import path
from subprocess import run

import pytest

from nlp_architect import LIBRARY_ROOT, LIBRARY_PATH
from nlp_architect.version import NLP_ARCHITECT_VERSION


class DocsCommand(object):
    cmd_name = 'doc'
    docs_source = path.join(LIBRARY_ROOT, 'doc')

    def __init__(self, subparsers):
        parser = subparsers.add_parser(DocsCommand.cmd_name,
                                       description='Build documentation',
                                       help='Build documentation')
        parser.set_defaults(func=DocsCommand.run_docs)
        self.parser = parser

    @staticmethod
    def run_docs(args):
        base_cmd = 'make -C {}'.format(DocsCommand.docs_source)
        print('Re-building documentation')
        run(base_cmd + ' clean', shell=True, check=True)
        run(base_cmd + ' html', shell=True, check=True)
        print('Documentation built in: {}'.format(path.join(DocsCommand.docs_source,
                                                            'build', 'html')))
        print('To view documents open your browser to: http://localhost:8000')

        class HTTPHandler(SimpleHTTPRequestHandler):
            def translate_path(self, path):
                path = SimpleHTTPRequestHandler.translate_path(self, path)
                relpath = os.path.relpath(path, os.getcwd())
                fullpath = os.path.join(self.server.base_path, relpath)
                return fullpath

        class HTTPServer(BaseHTTPServer):
            def __init__(self, base_path, server_address, RequestHandlerClass=HTTPHandler):
                self.base_path = base_path
                BaseHTTPServer.__init__(self, server_address, RequestHandlerClass)

        web_dir = os.path.join(path.join(DocsCommand.docs_source, 'build', 'html'))
        httpd = HTTPServer(web_dir, ("", 8000))
        httpd.serve_forever()


class StyleCommand(object):
    cmd_name = 'style'
    check_dirs = ['examples/',
                  'nlp_architect/',
                  'tests/',
                  ]
    files_to_check = [path.join(LIBRARY_ROOT, f) for f in check_dirs]

    def __init__(self, subparsers):
        parser = subparsers.add_parser(StyleCommand.cmd_name,
                                       description='Run style checks on NLP Architect code',
                                       help='Run style checks on NLP Architect code')
        parser.add_argument('--only-flake', default=False, action='store_true',
                            help='Run only Flake8 style check')
        parser.add_argument('--only-pylint', default=False, action='store_true',
                            help='Run only Pylint style check')
        parser.set_defaults(func=StyleCommand.run_check)
        self.parser = parser

    @staticmethod
    def run_check(args):
        rc = 0
        run_all = not args.only_flake and not args.only_pylint
        if args.only_flake or run_all:
            f_err = StyleCommand.run_flake()
            if int(f_err) > 0:
                print('Flake8 errors found. RC={}'.format(f_err))
                rc = 1
        if args.only_pylint or run_all:
            p_err = StyleCommand.run_pylint()
            if 0 < int(p_err) < 3:
                print('pylint errors found. RC={}'.format(p_err))
                rc = 1
        if rc > 0:
            exit(1)

    @staticmethod
    def run_flake():
        print('Running flake8 ...\n')
        flake8_config = path.join(LIBRARY_ROOT, 'setup.cfg')
        cmd = 'flake8 {} --config={}'.format(' '.join(StyleCommand.files_to_check), flake8_config)
        cmd_out = run(cmd, shell=True, check=True)
        return cmd_out.returncode

    @staticmethod
    def run_pylint():
        print('Running pylint ...\n')
        pylint_config = path.join(LIBRARY_ROOT, 'pylintrc')
        cmd = 'pylint {} --rcfile {}'.format(' '.join(StyleCommand.files_to_check), pylint_config)
        cmd_out = run(cmd, shell=True, check=False)
        return cmd_out.returncode


class TestCommand(object):
    cmd_name = 'test'

    def __init__(self, subparsers):
        parser = subparsers.add_parser(TestCommand.cmd_name,
                                       description='Run NLP Architect tests',
                                       help='Run NLP Architect tests')
        parser.add_argument('-f', type=str, help='test filename (runs test only of this file)')
        parser.set_defaults(func=TestCommand.run_tests)
        self.parser = parser

    @staticmethod
    def run_tests(args):
        # run all tests
        print('\nrunning NLP Architect tests ...')
        tests_dir = path.join(LIBRARY_ROOT, 'tests')
        tests = None
        if args.f:
            specific_test_file = args.f
            test_path = path.join(os.getcwd(), specific_test_file)
            if path.exists(test_path):
                tests = test_path
        else:
            tests = tests_dir
        if tests:
            TestCommand._prepare_tests()
            exit_code = pytest.main([tests, '-rs', '-vv'])
            print('tests exit code={}'.format(exit_code))
            if int(exit_code) != 0:
                exit(1)

    @staticmethod
    def _prepare_tests():
        import spacy
        from spacy.cli.download import download as spacy_download
        try:
            spacy.load('en')
        except OSError:
            spacy_download('en')

        from nlp_architect.api.machine_comprehension_api import MachineComprehensionApi
        from nlp_architect.api.intent_extraction_api import IntentExtractionApi
        from nlp_architect.api.ner_api import NerApi
        NerApi(prompt=False)
        IntentExtractionApi(prompt=False)
        MachineComprehensionApi(prompt=False).download_model()


class ServerCommand(object):
    cmd_name = 'server'

    def __init__(self, subparsers):
        parser = subparsers.add_parser(ServerCommand.cmd_name,
                                       description='Run NLP Architect server and demo UI',
                                       help='Run NLP Architect server and demo UI')
        parser.add_argument('-p', '--port', type=int, default=8080, help='server port')
        parser.set_defaults(func=ServerCommand.run_server)
        self.parser = parser

    @staticmethod
    def run_server(args):
        port = args.port
        serve_file = path.join(LIBRARY_PATH, 'server', 'serve.py')
        cmd_str = 'hug -p {} -f {}'.format(port, serve_file)
        run(cmd_str, shell=True, check=True)


# sub commands list
sub_commands = [
    TestCommand,
    StyleCommand,
    DocsCommand,
    ServerCommand,
]


def main():
    prog = 'nlp_architect'
    parser = argparse.ArgumentParser(description='NLP Architect runner', prog=prog)
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s v{}'.format(NLP_ARCHITECT_VERSION))

    subparsers = parser.add_subparsers(title='commands', metavar='')
    for command in sub_commands:
        command(subparsers)
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
