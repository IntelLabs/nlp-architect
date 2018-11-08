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
import subprocess
from http.server import HTTPServer as BaseHTTPServer, SimpleHTTPRequestHandler
from os import path

import pytest

from nlp_architect import LIBRARY_ROOT
from nlp_architect.version import nlp_architect_version


def run_shell_cmd(cmd, stdout=subprocess.PIPE):
    p = subprocess.Popen([cmd], stdout=stdout, stderr=subprocess.STDOUT, shell=True)
    out = p.communicate()[0].decode('UTF-8')
    ret_code = p.returncode
    return out, ret_code


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
        run_shell_cmd(base_cmd + ' clean')
        out, _ = run_shell_cmd(base_cmd + ' html')
        print(out)
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
                  'server/',
                  'tests/',
                  'solutions/'
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
        out, ret = run_shell_cmd(cmd)
        print(out)
        return ret

    @staticmethod
    def run_pylint():
        print('Running pylint ...\n')
        pylint_config = path.join(LIBRARY_ROOT, 'pylintrc')
        cmd = 'pylint {} --rcfile {}'.format(' '.join(StyleCommand.files_to_check), pylint_config)
        out, ret = run_shell_cmd(cmd)
        print(out)
        return ret


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


class DemoServerCommand(object):
    cmd_name = 'demo'

    def __init__(self, subparsers):
        parser = subparsers.add_parser(DemoServerCommand.cmd_name,
                                       description='Run NLP Architect server and demo UI',
                                       help='Run NLP Architect server and demo UI')
        parser.add_argument('-p', '--port', type=int, default=8080, help='server port')
        parser.set_defaults(func=DemoServerCommand.run_server)
        self.parser = parser

    @staticmethod
    def run_server(args):
        port = args.port
        serve_file = path.join(LIBRARY_ROOT, 'server/serve.py')
        cmd_str = 'hug -p {} -f {}'.format(port, serve_file)
        run_shell_cmd(cmd_str, stdout=False)


# sub commands list
sub_commands = [
    TestCommand,
    StyleCommand,
    DocsCommand,
    DemoServerCommand,
]


def main():
    prog = 'nlp_architect'
    parser = argparse.ArgumentParser(description='NLP Architect runner', prog=prog)
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s v{}'.format(nlp_architect_version()))

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
