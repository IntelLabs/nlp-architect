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

import io
import pathlib
import pickle
from os import path, walk

from spacy import load as spacy_load

from models.bist.bmstparser import mstlstm
from models.bist.bmstparser.parser_utils import ConllEntry
from ai_lab_nlp.pipelines.spacy_bist.utils import download_file, unzip_file
from ai_lab_nlp.utils.parsed_document import ParsedDocument

loaded_models = {}


class SpacyBISTParser:
    """
    Args:
        verbose (:obj:`bool`, optional): Controls output verbosity.
            Defaults to False.
        spacy_model (:obj:`str`, optional): Spacy model to use (see
            https://spacy.io/api/top-level#spacy.load). Defaults to 'en' model.
        bist_model (:obj:`str`, optional): Path to a .model file to load.
            Defaults to 'spacy_bist/pretrained/bist.model'.
    """
    dir = path.dirname(path.realpath(__file__))
    pretrained = path.join(dir, 'bist-pretrained', 'bist.model')

    def __init__(self, verbose=False, spacy_model='en', bist_model=None):
        if not bist_model:
            print("Using pre-trained BIST model.")
            download_pretrained_model()

        self.verbose = verbose
        self.bist_parser = \
            load_bist_model(bist_model if bist_model else SpacyBISTParser.pretrained)
        print('Loading Spacy annotator...')
        self.spacy_annotator = spacy_load(spacy_model)

    def to_conll(self, doc_text):
        """Converts a document to CoNLL format.

        Args:
            doc_text (str): raw document text

        Yields:
            list of ConllEntry: The sentences in the document in CoNLL format.
        """
        for sentence in self.spacy_annotator(doc_text).sents:
            sentence_conll = [ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_',
                                         -1, 'rroot', '_', '_')]
            i_tok = 0
            for tok in sentence:
                if self.verbose:
                    print(tok.text + '\t' + tok.tag_)

                if not tok.is_space:
                    pos = tok.tag_
                    text = tok.text

                    if text != '-' or pos != 'HYPH':
                        pos = normalize_pos(pos, text)
                        token_conll = ConllEntry(i_tok + 1, text, tok.lemma_, pos, pos,
                                                 tok.ent_type_, -1, '_', '_', tok.idx)
                        sentence_conll.append(token_conll)
                        i_tok += 1

            if self.verbose:
                print('-----------------------\ninput conll form:')
                for entry in sentence_conll:
                    print(str(entry.id) + '\t' + entry.form + '\t' + entry.pos + '\t')
            yield sentence_conll

    def parse_dir(self, input_dir, out_dir=None, show_tok=True, show_doc=True):
        """Parses a directory of documents.

        Args:
            input_dir (str)
            out_dir (:obj:`str`, optional): If specified, the output will
                also be written to this path.
            show_tok (:obj:`bool`, optional): Specifies whether to include
                token text in output.
            show_doc (:obj:`bool`, optional): Specifies whether to include
                document text in output.

        Returns:
            (:obj:`list` of :obj:`ParsedDocument`)
        """
        return [_ for _ in self.parse_dir_gen(input_dir, out_dir, show_tok, show_doc)]

    def parse_dir_gen(self, input_dir, out_dir=None, show_tok=True, show_doc=True):
        """Parse a directory of raw text documents, one by one.

        Args:
            input_dir (str)
            out_dir (:obj:`str`, optional): If specified, the output will
                also be written to this path.
            show_tok (:obj:`bool`, optional): Specifies whether to include
                token text in output.
            show_doc (:obj:`bool`, optional): Specifies whether to include
                document text in output.

        Yields:
            ParsedDocument: the annotated document.
        """
        for i, (filename, file_contents) in enumerate(walk_directory(input_dir)):
            print('Parsing document No. ' + str(i) + ': ' + filename)
            parsed_doc = self.parse(file_contents, show_tok, show_doc)

            if out_dir:
                pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
                out_path = out_dir + '/' + filename + '.json'
                print('Dumping parsed document to ' + out_path)
                with io.open(out_path, 'w') as file:
                    file.write(parsed_doc.pretty_json())
            yield parsed_doc

    def parse(self, doc_text, show_tok=True, show_doc=True):
        """Parse a raw text document.

        Args:
            doc_text (str)
            show_tok (:obj:`bool`, optional): Specifies whether to include
                token text in output.
            show_doc (:obj:`bool`, optional): Specifies whether to include
                document text in output.

        Yields:
            ParsedDocument: The annotated document.
        """
        doc_conll = self.to_conll(doc_text)
        parsed_doc = ParsedDocument()

        if show_doc:
            parsed_doc.doc_text = doc_text

        for sent_conll in self.bist_parser.predict(conll=doc_conll):
            parsed_sent = []
            conj_governors = {'and': set(), 'or': set()}

            for tok in sent_conll:
                gov_id = int(tok.pred_parent_id)
                rel = tok.pred_relation

                if tok.form != '*root*':
                    if tok.form.lower() == 'and':
                        conj_governors['and'].add(gov_id)
                    if tok.form.lower() == 'or':
                        conj_governors['or'].add(gov_id)

                    if rel == 'conj':
                        if gov_id in conj_governors['and']:
                            rel += '_and'
                        if gov_id in conj_governors['or']:
                            rel += '_or'

                    parsed_tok = {'start': tok.misc, 'len': len(tok.form),
                                  'pos': tok.pos, 'ner': tok.feats,
                                  'lemma': tok.lemma, 'gov': gov_id - 1,
                                  'rel': rel}

                    if show_tok:
                        parsed_tok['text'] = tok.form
                    parsed_sent.append(parsed_tok)
            if parsed_sent:
                parsed_doc.sentences.append(parsed_sent)
        return parsed_doc

    def inference(self, input_dir=None, out_dir=None, doc_text=None, show_tok=True, show_doc=True):
        """Run inference on a document text or a directory of documents.

        Args:
            input_dir (:obj:`str`, optional)
            out_dir (:obj:`str`, optional)
            doc_text (:obj:`str`, optional)
            show_tok (:obj:`bool`, optional): Specifies whether to include
                token text in output.
            show_doc (:obj:`bool`, optional): Specifies whether to include
                document text in output.

        Returns:
            A ParsedDocument or list of ParsedDocument, depending on the input.
        """
        if (input_dir and doc_text) or (not input_dir and not doc_text):
            print('Usage: Please specify either an input_dir or a doc_text')
            return None
        if input_dir:
            return self.parse_dir(input_dir, out_dir)
        return self.parse(doc_text, show_tok, show_doc)


def load_bist_model(model_path):
    """Loads and initializes a BIST LSTM model from file."""
    if model_path not in loaded_models:
        print('Loading BIST Parser...')
        params_path = \
            path.join(str(pathlib.Path(path.abspath(model_path)).parent), 'params.pickle')
        with io.open(params_path, 'rb') as file:
            params = pickle.load(file)
        bist_parser = mstlstm.MSTParserLSTM(*params)
        bist_parser.load(model_path)
        loaded_models[model_path] = bist_parser
    return loaded_models[model_path]


def download_pretrained_model():
    """Downloads the pre-trained BIST model if non-existent."""
    if not path.exists(path.join(SpacyBISTParser.dir, 'bist-pretrained', 'bist.model')):
        print('Downloading pre-trained BIST model...')
        download_file('https://s3-us-west-1.amazonaws.com/nervana-modelzoo/parse/',
                      'bist-pretrained.zip', path.join(SpacyBISTParser.dir, 'bist-pretrained.zip'))
        print('Unzipping...')
        unzip_file(path.join(SpacyBISTParser.dir, 'bist-pretrained.zip'))
        print('Done.')


def normalize_pos(pos, text):
    """Converts a Spacy part-of-speech tag to a Penn Treebank part-of-speech tag."""
    norm_pos = pos
    if text == '...':
        norm_pos = ':'
    elif text == '*':
        norm_pos = 'SYM'
    elif pos == 'AFX':
        norm_pos = 'JJ'
    elif pos == 'ADD':
        norm_pos = 'NN'
    elif text != pos and text in [',', '.', ":", '``', '-RRB-', '-LRB-']:
        norm_pos = text
    elif pos in ['NFP', 'HYPH', 'XX']:
        norm_pos = 'SYM'
    return norm_pos


def walk_directory(directory):
    """Iterates a directory's text files and their contents."""
    for dir_path, _, filenames in walk(directory):
        for filename in filenames:
            file_path = path.join(dir_path, filename)
            if path.isfile(file_path) and not filename.startswith('.'):
                with io.open(file_path, 'r', encoding='utf-8') as file:
                    print('Reading ' + filename)
                    doc_text = file.read()
                    yield filename, doc_text