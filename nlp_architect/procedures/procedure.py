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
import abc


class Procedure:
    def __init__(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        self.parser = parser

    @staticmethod
    @abc.abstractmethod
    def add_arguments(parser: argparse.ArgumentParser):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def run_procedure(args):
        raise NotImplementedError

    @classmethod
    def run(cls):
        parser = argparse.ArgumentParser()
        cls.add_arguments(parser)
        cls.run_procedure(parser.parse_args())
