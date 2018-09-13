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
# pylint: disable=redefined-outer-name
import pytest

from nlp_architect.utils.text import try_to_load_spacy
from nlp_architect.pipelines.spacy_bist import SpacyBISTParser

if not try_to_load_spacy('en'):
    pytest.skip("\n\nSkipping test_spacy_bist.py. Reason: 'spacy en' model not installed. "
                "Please see https://spacy.io/models/ for installation instructions.\n"
                "The terms and conditions of the data set and/or model license apply.\n"
                "Intel does not grant any rights to the data and/or model files.\n",
                allow_module_level=True)


class TestData:
    """Test cases for the test functions below."""
    output_structure = ["This is a single-sentence document",
                        "This is a document... This is the second sentence"]

    sentence_breaking = [("This is a single-sentence document", 1),
                         ("This is a document... This is the second sentence", 2)]

    dependencies = \
        [("i don't have other assistance",
          [[('nsubj', 3), ('aux', 3), ('neg', 3), ('root', -1), ('amod', 5), ('dobj', 3)]])]

    pos_tags = ["He's the best player in NBA history"]


class Fixtures:
    default_parser = SpacyBISTParser()

    ptb_pos_tags = {'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
                    'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
                    'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                    'VBZ', 'WDT', 'WP', 'WP$', 'WRB'}

    token_label_types = {'start': int, 'len': int, 'pos': str, 'ner': str, 'lemma': str,
                         'gov': int, 'rel': str}


@pytest.mark.parametrize('show_tok', [True, False])
@pytest.mark.parametrize('show_doc', [True, False])
@pytest.mark.parametrize('text', TestData.output_structure)
def test_output_structure(parser, text, show_tok, show_doc, token_label_types):
    """Test that the output object structure hasn't changed.

    Args:
        parser (SpacyBistParser)
        text (str): Input test case.
        show_tok (bool): Specifies whether to include token text in output.
        show_doc (bool): Specifies whether to include document text in output.
        token_label_types (dict): Mapping of label names to their type.
    """
    parsed_doc = parser.parse(doc_text=text, show_tok=show_tok, show_doc=show_doc)
    assert isinstance(parsed_doc.sentences, list)
    assert isinstance(parsed_doc.doc_text, str) if show_doc else not parsed_doc.doc_text
    for sentence in parsed_doc:
        for token in sentence:
            assert isinstance(token['text'], str) if show_tok else 'text' not in token
            for label, label_type in token_label_types.items():
                assert isinstance(token.get(label), label_type)


@pytest.mark.parametrize('method_name', ['parse'])
@pytest.mark.parametrize('text, sent_count', TestData.sentence_breaking)
def test_sentence_breaking(parser, text, sent_count, method_name):
    """Test that documents are broken into the expected number of sentences.

    Args:
        parser (SpacyBistParser)
        text (str): Input test case.
        sent_count (int): Expected number of sentences in `text`.
        method_name (str): Parse method to test.
    """
    parse_method = getattr(parser, method_name)
    parsed_doc = parse_method(doc_text=text)
    assert len(parsed_doc.sentences) == sent_count


@pytest.mark.parametrize('method_name', ['parse'])
@pytest.mark.parametrize('text, deps', TestData.dependencies)
def test_dependencies(parser, text, deps, method_name):
    """Test that dependencies are predicted correctly.

    Args:
        parser (SpacyBistParser)
        text (str): Input test case.
        deps (list of list of tuples): Expected dependencies in `text`.
        method_name (str): Parse method to test.
    """
    parse_method = getattr(parser, method_name)
    parsed_doc = parse_method(doc_text=text)
    for i_sentence, sentence in enumerate(parsed_doc):
        for i_token, token in enumerate(sentence):
            assert (token['rel'], token['gov']) == deps[i_sentence][i_token]


@pytest.mark.parametrize('method_name', ['parse'])
@pytest.mark.parametrize('text', TestData.pos_tags)
def test_pos_tag(parser, text, method_name, ptb_pos_tags):
    """Tests that produced POS tags are valid PTB POS tags.

    Args:
        parser (SpacyBistParser)
        text (str): Input test case.
        method_name (str): Parse method to test.
        ptb_pos_tags (set of str): Valid PTB POS tags.
    """
    parse_method = getattr(parser, method_name)
    parsed_doc = parse_method(doc_text=text)
    assert all([token['pos'] in ptb_pos_tags for sent in parsed_doc for token in sent])


@pytest.fixture
def parser():
    return Fixtures.default_parser


@pytest.fixture
def ptb_pos_tags():
    return Fixtures.ptb_pos_tags


@pytest.fixture
def token_label_types():
    return Fixtures.token_label_types
