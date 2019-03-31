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
import shutil
import subprocess
from http.server import HTTPServer as BaseHTTPServer, SimpleHTTPRequestHandler
from subprocess import run

import pytest

from nlp_architect import LIBRARY_ROOT, LIBRARY_PATH, LIBRARY_OUT
from nlp_architect.utils import ansi2html
from nlp_architect.version import NLP_ARCHITECT_VERSION


def run_cmd(command):
    return run(command.split(), shell=False)


class DocsCommand(object):
    cmd_name = 'doc'
    docs_source = LIBRARY_ROOT / 'doc'

    def __init__(self, subparsers):
        parser = subparsers.add_parser(DocsCommand.cmd_name,
                                       description='Build documentation',
                                       help='Build documentation')
        parser.set_defaults(func=DocsCommand.run_docs)
        self.parser = parser

    @staticmethod
    def run_docs(_):
        base_cmd = 'make -C {}'.format(DocsCommand.docs_source)
        print('Re-building documentation')
        run_cmd(base_cmd + ' clean')
        run_cmd(base_cmd + ' html')
        print('Documentation built in: {}'.format(DocsCommand.docs_source / 'build' / 'html'))
        print('To view documents point your browser to: http://localhost:8000')

        class HTTPHandler(SimpleHTTPRequestHandler):
            def translate_path(self, path):
                path = SimpleHTTPRequestHandler.translate_path(self, path)
                relpath = os.path.relpath(path, os.getcwd())
                fullpath = os.path.join(self.server.base_path, relpath)
                return fullpath

        class HTTPServer(BaseHTTPServer):
            def __init__(self, base_path, server_address, request_handler_class=HTTPHandler):
                self.base_path = base_path
                BaseHTTPServer.__init__(self, server_address, request_handler_class)

        web_dir = DocsCommand.docs_source / 'build' / 'html'
        httpd = HTTPServer(web_dir, ("", 8000))
        httpd.serve_forever()


class StyleCommand(object):
    cmd_name = 'style'
    check_dirs = [
        'examples',
        'nlp_architect',
        'tests'
    ]
    files_to_check = [str(LIBRARY_ROOT / f) for f in check_dirs]

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
            if int(p_err) != 0:
                print('pylint errors found. RC={}'.format(p_err))
                rc = 1
        if rc > 0:
            exit(1)

    @staticmethod
    def run_flake():
        print('Running flake8 ...\n')
        flake8_config = str(LIBRARY_ROOT / 'setup.cfg')
        os.makedirs(LIBRARY_OUT, exist_ok=True)
        flake8_out = str(LIBRARY_OUT / 'flake8.txt')
        flake8_html_out = str(LIBRARY_OUT / 'flake_html')
        print('HHH:' + flake8_html_out)
        try:
            os.remove(flake8_out)
            shutil.rmtree(flake8_html_out, ignore_errors=True)
        except OSError:
            pass
        cmd = 'flake8 {} --config={} --output-file {}' \
            .format(' '.join(StyleCommand.files_to_check), flake8_config, flake8_out)
        cmd_out = run_cmd(cmd)

        cmd_html = 'flake8 {} --config={} --format=html --htmldir={}' \
            .format(' '.join(StyleCommand.files_to_check), flake8_config, flake8_html_out)

        run_cmd(cmd_html)
        print('To view flake8 results in html, point your web browser to: \n{}\n'
              .format(flake8_html_out + '/index.html'))

        return cmd_out.returncode

    @staticmethod
    def run_pylint():
        print('Running pylint ...\n')
        pylint_config = LIBRARY_ROOT / 'pylintrc'
        os.makedirs(LIBRARY_OUT, exist_ok=True)
        pylint_out = str(LIBRARY_OUT / 'pylint.txt')
        html_out = str(LIBRARY_OUT / 'pylint.html')

        cmd = 'pylint -j 4 {} --rcfile {} --score=n'\
            .format(' '.join(StyleCommand.files_to_check), pylint_config)
        cmd_tee = 'tee {}'.format(pylint_out)
        ps = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        subprocess.run(cmd_tee.split(), stdin=ps.stdout)
        ret_code = 1 if os.stat(pylint_out).st_size else 0

        ansi2html.run(pylint_out, html_out)
        print('To view pylint results in html, point your web browser to: \n{}\n'.format(html_out))
        return ret_code


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
        tests_dir = str(LIBRARY_ROOT / 'tests')
        tests = None
        if args.f:
            specific_test_file = args.f
            test_path = os.path.join(os.getcwd(), specific_test_file)
            if os.path.exists(test_path):
                tests = test_path
        else:
            tests = tests_dir
        if tests:
            TestCommand._prepare_tests()
            exit_code = pytest.main([tests,
                                     '-rs',
                                     '-vv',
                                     '--cov=nlp_architect',
                                     '--junit-xml=pytest_unit.xml'])
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
        serve_file = LIBRARY_PATH / 'server' / 'serve.py'
        cmd_str = 'hug -p {} -f {}'.format(port, serve_file)
        run_cmd(cmd_str)


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
