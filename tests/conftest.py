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
from __future__ import unicode_literals, print_function, division, \
    absolute_import

from ai_lab_nlp.pipelines.spacy_bist.parser import SpacyBISTParser
import pytest


class Fixtures:
    """
    Pytest fixtures for all tests.
    """
    default_parser = SpacyBISTParser()

    ptb_pos_tags = {'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
                    'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
                    'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                    'VBZ', 'WDT', 'WP', 'WP$', 'WRB'}

    token_label_types = {'start': int, 'len': int, 'pos': str, 'ner': str, 'lemma': str,
                         'gov': int, 'rel': str}

    parse_str_methods = ['inference', 'parse']


@pytest.fixture
def parser():
    return Fixtures.default_parser


@pytest.fixture
def ptb_pos_tags():
    return Fixtures.ptb_pos_tags


@pytest.fixture
def token_label_types():
    return Fixtures.token_label_types


@pytest.fixture
def parse_str_methods():
    return Fixtures.parse_str_methods
