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
import csv
import json
from os import walk, path, makedirs, PathLike, listdir
from os.path import join, isfile, isdir
from pathlib import Path

import numpy as np

from nlp_architect.common.core_nlp_doc import CoreNLPDoc
from nlp_architect.models.absa import INFERENCE_LEXICONS, TRAIN_LEXICONS
from nlp_architect.models.absa.inference.data_types import LexiconElement, Polarity
from nlp_architect.models.absa.train.data_types import OpinionTerm, string_list_headers
from nlp_architect.utils.io import download_unlicensed_file


def _download_pretrained_rerank_model(rerank_model_full_path):
    rerank_model_dir = path.dirname(rerank_model_full_path)
    if not path.isfile(rerank_model_full_path):
        makedirs(rerank_model_dir, exist_ok=True)
        print("dowloading pre-trained reranking model..")
        download_unlicensed_file('https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/'
                                 'absa/', 'rerank_model.h5', rerank_model_full_path)
    return rerank_model_full_path


def _walk_directory(directory):
    """Iterates a directory's text files and their contents."""
    for dir_path, _, filenames in walk(directory):
        for filename in filenames:
            file_path = join(dir_path, filename)
            if isfile(file_path) and not filename.startswith('.'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    print('Reading ' + filename)
                    doc_text = file.read()
                    yield filename, doc_text


def parse_docs(parser, docs, out_dir=None, show_tok=True, show_doc=True):
    """Parse raw documents in the form of text files in a directory or lines in a text file.

    Args:
        parser (SpacyBISTParser)
        docs (str or PosixPath)
        out_dir (PosixPath): If specified, the output will also be written to this path.
        show_tok (bool, optional): Specifies whether to include token text in output.
        show_doc (bool, optional): Specifies whether to include document text in output.

    Returns:
        (list of CoreNLPDoc)
    """
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    params = parser, Path(docs), out_dir, show_tok, show_doc
    parsed_docs = list(parse_dir(*params) if isdir(docs) else parse_txt(*params))
    total_parsed = np.sum([len(doc) for doc in parsed_docs])
    return parsed_docs, total_parsed


def parse_txt(parser, txt_path, out_dir=None, show_tok=True, show_doc=True):
    """Parse raw documents in the form of lines in a text file.

    Args:
        parser (SpacyBISTParser)
        txt_path (PosixPath)
        out_dir (PosixPath): If specified, the output will also be written to this path.
        show_tok (bool, optional): Specifies whether to include token text in output.
        show_doc (bool, optional): Specifies whether to include document text in output.

    Yields:
        CoreNLPDoc: the annotated document.
    """
    with open(txt_path, encoding='utf-8') as f:
        for i, doc_text in enumerate(f):
            print('Parsing document no. {}'.format(i + 1))
            parsed_doc = parser.parse(doc_text.rstrip('\n'), show_tok, show_doc)

            if out_dir:
                out_path = out_dir / (str(i + 1) + '.json')
                print('Dumping parsed document to {}'.format(out_path))
                with open(out_path, 'w', encoding='utf-8') as doc_file:
                    doc_file.write(parsed_doc.pretty_json())

            yield parsed_doc


def parse_dir(parser, input_dir, out_dir=None, show_tok=True, show_doc=True):
    """Parse a directory of raw text documents, one by one.

    Args:
        parser (SpacyBISTParser)
        input_dir (PosixPath)
        out_dir (PosixPath): If specified, the output will also be written to this path.
        show_tok (bool, optional): Specifies whether to include token text in output.
        show_doc (bool, optional): Specifies whether to include document text in output.

    Yields:
        CoreNLPDoc: the annotated document.
    """
    for i, (filename, file_contents) in enumerate(_walk_directory(input_dir)):
        print('Parsing document No. {}: {}'.format(str(i), filename))
        parsed_doc = parser.parse(file_contents, show_tok, show_doc)

        if out_dir:
            out_path = out_dir / (filename + '.json')
            print('Dumping parsed document to {}'.format(out_path))
            with open(out_path, 'w') as file:
                file.write(parsed_doc.pretty_json())
        yield parsed_doc


def _read_lexicon_from_csv(lexicon_path: str) -> dict:
    """Read a lexicon from a CSV file.

    Returns: Dictionary of LexiconElements, each LexiconElement presents a row.
    """
    lexicon = {}
    with open(INFERENCE_LEXICONS / lexicon_path, newline='', encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        for row in reader:
            try:
                lexicon[row[0]] = LexiconElement(term=row[0], score=row[1], polarity=None,
                                                 is_acquired=None, position=row[2])
            except IndexError:
                lexicon[row[0]] = LexiconElement(term=row[0], score=row[1], polarity=None,
                                                 is_acquired=None, position=None)
    return lexicon


def load_opinion_lex(file_name) -> dict:
    """Read opinion lexicon from CSV file.

    Returns: Dictionary of LexiconElements, each LexiconElement presents a row.
    """
    lexicon = {}
    with open(file_name, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)
        for row in reader:
            term, score, polarity, is_acquired = row[0], row[1], row[2], row[3]
            score = float(score)
            # ignore terms with low score
            if score >= 0.5 and polarity in (Polarity.POS.value, Polarity.NEG.value):
                lexicon[term] = \
                    LexiconElement(term.lower(),
                                   score if polarity == Polarity.POS.value else -score, polarity,
                                   is_acquired)
    return lexicon


def _load_aspect_lexicon(file_name):
    """Read aspect lexicon from CSV file.

    Returns: Dictionary of LexiconElements, each LexiconElement presents a row.
    """
    lexicon = []
    with open(file_name, 'r', newline='', encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        for row in reader:
            lexicon.append(LexiconElement(row))
    return lexicon


def _load_parsed_docs_from_dir(directory: str or PathLike):
    """Read all file in directory.

    Args:
        directory (PathLike): path
    """
    res = {}
    for file_name in listdir(directory):
        if file_name.endswith('.txt') or file_name.endswith('.json'):
            with open(Path(directory) / file_name) as f:
                content = f.read()
                res[file_name] = json.loads(content, object_hook=CoreNLPDoc.decoder)
    return res


def _write_table(table, filename):
    """Write table as csv to file system.

    Args:
        table (list): table to be printed as list
        filename (str): file name
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for row in table:
            writer.writerow(row)


def _write_final_lex(dictionary, file_name):
    """Write generated lex as csv to file.

    Args:
        dictionary (list): list of filtered terms
        file_name (str): file name
    """
    candidate_terms = [["#"] + string_list_headers()]
    term_num = 1
    for candidate_term in dictionary:
        term_row = [int(term_num)] + candidate_term.as_string_list()
        candidate_terms.append(term_row)
        term_num += 1
    _write_table(candidate_terms, file_name)


def _write_generic_sentiment_terms(dictionary, file_name):
    """Write generic sentiment terms as csv to file system."""
    generic_terms = [["Term", "Score", "Polarity"]]
    for generic_term in dictionary.values():
        generic_terms.append(
            [str(generic_term), "1.0", generic_term.polarity.name])
    _write_table(generic_terms, file_name)


def _load_lex_as_list_from_csv(file_name):
    """Load lexicon as list.

    Args:
        file_name (str): input csv file name
    """
    lexicon_table = []
    with open(TRAIN_LEXICONS / file_name, 'r') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if reader is None:
            print("file name is None")
            return lexicon_table
        for row in reader:
            term = row['Term'].strip()
            lexicon_table.append(term)

        return lexicon_table


def read_generic_lex_from_file(file_name):
    """Read generic opinion lex for term acquisition.

    Args:
        file_name (string): name of csv file
    """
    with open(file_name) as f:
        reader = csv.DictReader(f)
        dict_list = {}
        for row in reader:
            if row["UsedForAcquisition"] == 'Y':
                key_term = row["Term"] if len(row) > 0 else ""
                terms = []
                if len(row) > 0:
                    terms = key_term.split()
                polarity = Polarity[row["Polarity"]]
                dict_list[key_term] = OpinionTerm(terms, polarity)
    return dict_list


def _read_generic_lex_for_similarity(file_name):
    """Read generic opinion terms for similarity calc from csv file.

    Args:
        file_name (string): name of csv file
    """
    with open(file_name) as f:
        reader = csv.DictReader(f)
        dict_list = {}
        for row in reader:
            if row["UsedForReranking"] == 'Y':
                key_term = row["Term"] if len(row) > 0 else ""
                polarity = row["Polarity"]
                dict_list[key_term] = polarity
    return dict_list
