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
from os import path, remove, makedirs

from nlp_architect.common.core_nlp_doc import CoreNLPDoc
from nlp_architect.data.conll import ConllEntry
from nlp_architect.models.bist_parser import BISTModel
from nlp_architect import LIBRARY_OUT
from nlp_architect.utils.io import download_unlicensed_file, uncompress_file
from nlp_architect.utils.io import validate
from nlp_architect.utils.text import SpacyInstance


class SpacyBISTParser(object):
    """Main class which handles parsing with Spacy-BIST parser.

    Args:
        verbose (bool, optional): Controls output verbosity.
        spacy_model (str, optional): Spacy model to use
        (see https://spacy.io/api/top-level#spacy.load).
        bist_model (str, optional): Path to a .model file to load. Defaults pre-trained model'.
    """
    dir = LIBRARY_OUT / 'bist-pretrained'
    _pretrained = dir / 'bist.model'

    def __init__(self, verbose=False, spacy_model='en', bist_model=None):
        validate((verbose, bool), (spacy_model, str, 0, 1000),
                 (bist_model, (type(None), str), 0, 1000))
        if not bist_model:
            print("Using pre-trained BIST model.")
            _download_pretrained_model()
            bist_model = SpacyBISTParser._pretrained

        self.verbose = verbose
        self.bist_parser = BISTModel()
        self.bist_parser.load(bist_model if bist_model else SpacyBISTParser._pretrained)
        self.spacy_parser = SpacyInstance(spacy_model,
                                          disable=['ner', 'vectors', 'textcat']).parser

    def to_conll(self, doc_text):
        """Converts a document to CoNLL format with spacy POS tags.

        Args:
            doc_text (str): raw document text.

        Yields:
            list of ConllEntry: The next sentence in the document in CoNLL format.
        """
        validate((doc_text, str))
        for sentence in self.spacy_parser(doc_text).sents:
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
                        pos = _spacy_pos_to_ptb(pos, text)
                        token_conll = ConllEntry(i_tok + 1, text, tok.lemma_, pos, pos,
                                                 tok.ent_type_, -1, '_', '_', tok.idx)
                        sentence_conll.append(token_conll)
                        i_tok += 1

            if self.verbose:
                print('-----------------------\ninput conll form:')
                for entry in sentence_conll:
                    print(str(entry.id) + '\t' + entry.form + '\t' + entry.pos + '\t')
            yield sentence_conll

    def parse(self, doc_text, show_tok=True, show_doc=True):
        """Parse a raw text document.

        Args:
            doc_text (str)
            show_tok (bool, optional): Specifies whether to include token text in output.
            show_doc (bool, optional): Specifies whether to include document text in output.

        Returns:
            CoreNLPDoc: The annotated document.
        """
        validate((doc_text, str), (show_tok, bool), (show_doc, bool))
        doc_conll = self.to_conll(doc_text)
        parsed_doc = CoreNLPDoc()

        if show_doc:
            parsed_doc.doc_text = doc_text

        for sent_conll in self.bist_parser.predict_conll(doc_conll):
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


def _download_pretrained_model():
    """Downloads the pre-trained BIST model if non-existent."""
    if not path.isfile(SpacyBISTParser.dir / 'bist.model'):
        print('Downloading pre-trained BIST model...')
        zip_path = SpacyBISTParser.dir / 'bist-pretrained.zip'
        makedirs(SpacyBISTParser.dir, exist_ok=True)
        download_unlicensed_file(
            'https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/dep_parse/',
            'bist-pretrained.zip', zip_path)
        print('Unzipping...')
        uncompress_file(zip_path, outpath=str(SpacyBISTParser.dir))
        remove(zip_path)
        print('Done.')


def _spacy_pos_to_ptb(pos, text):
    """
    Converts a Spacy part-of-speech tag to a Penn Treebank part-of-speech tag.

    Args:
        pos (str): Spacy POS tag.
        text (str): The token text.

    Returns:
        ptb_tag (str): Standard PTB POS tag.
    """
    validate((pos, str, 0, 30), (text, str, 0, 1000))
    ptb_tag = pos
    if text in ['...', '—']:
        ptb_tag = ':'
    elif text == '*':
        ptb_tag = 'SYM'
    elif pos == 'AFX':
        ptb_tag = 'JJ'
    elif pos == 'ADD':
        ptb_tag = 'NN'
    elif text != pos and text in [',', '.', ":", '``', '-RRB-', '-LRB-']:
        ptb_tag = text
    elif pos in ['NFP', 'HYPH', 'XX']:
        ptb_tag = 'SYM'
    return ptb_tag
