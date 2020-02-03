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
import sys
from os import walk, path, makedirs, PathLike, listdir
from os.path import join, isfile, isdir
from pathlib import Path
from typing import Union
from tqdm import tqdm
import numpy as np

from nlp_architect.common.core_nlp_doc import CoreNLPDoc
from nlp_architect.models.absa import INFERENCE_LEXICONS
from nlp_architect.models.absa.inference.data_types import LexiconElement, Polarity
from nlp_architect.models.absa.train.data_types import OpinionTerm
from nlp_architect.pipelines.spacy_bist import SpacyBISTParser
from nlp_architect.utils.io import download_unlicensed_file, line_count


def _download_pretrained_rerank_model(rerank_model_full_path):
    rerank_model_dir = path.dirname(rerank_model_full_path)
    if not path.isfile(rerank_model_full_path):
        makedirs(rerank_model_dir, exist_ok=True)
        print("dowloading pre-trained reranking model..")
        download_unlicensed_file(
            "https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/" "absa/",
            "rerank_model.h5",
            rerank_model_full_path,
        )
    return rerank_model_full_path


def _walk_directory(directory: Union[str, PathLike]):
    """Iterates a directory's text files and their contents."""
    for dir_path, _, filenames in walk(directory):
        for filename in filenames:
            file_path = join(dir_path, filename)
            if isfile(file_path) and not filename.startswith("."):
                with open(file_path, encoding="utf-8") as file:
                    doc_text = file.read()
                    yield filename, doc_text


def parse_docs(
    parser: SpacyBISTParser,
    docs: Union[str, PathLike],
    out_dir: Union[str, PathLike] = None,
    show_tok=True,
    show_doc=True,
):
    """Parse raw documents in the form of text files in a directory or lines in a text file.

    Args:
        parser (SpacyBISTParser)
        docs (str or PathLike)
        out_dir (str or PathLike): If specified, the output will also be written to this path.
        show_tok (bool, optional): Specifies whether to include token text in output.
        show_doc (bool, optional): Specifies whether to include document text in output.

    Returns:
        (list of CoreNLPDoc)
    """
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    params = parser, Path(docs), out_dir, show_tok, show_doc
    parsed_docs = list(parse_dir(*params) if isdir(docs) else parse_txt(*params))
    total_parsed = np.sum([len(doc) for doc in parsed_docs])
    return parsed_docs, total_parsed


def parse_txt(
    parser: SpacyBISTParser,
    txt_path: Union[str, PathLike],
    out_dir: Union[str, PathLike] = None,
    show_tok=True,
    show_doc=True,
):
    """Parse raw documents in the form of lines in a text file.

    Args:
        parser (SpacyBISTParser)
        txt_path (str or PathLike)
        out_dir (str or PathLike): If specified, the output will also be written to this path.
        show_tok (bool, optional): Specifies whether to include token text in output.
        show_doc (bool, optional): Specifies whether to include document text in output.

    Yields:
        CoreNLPDoc: the annotated document.
    """
    with open(txt_path, encoding="utf-8") as f:
        if out_dir:
            print("Writing parsed documents to {}".format(out_dir))
        for i, doc_text in enumerate(tqdm(f, total=line_count(txt_path), file=sys.stdout)):
            parsed_doc = parser.parse(doc_text.rstrip("\n"), show_tok, show_doc)

            if out_dir:
                out_path = Path(out_dir) / (str(i + 1) + ".json")
                with open(out_path, "w", encoding="utf-8") as doc_file:
                    doc_file.write(parsed_doc.pretty_json())
            yield parsed_doc


def parse_dir(
    parser,
    input_dir: Union[str, PathLike],
    out_dir: Union[str, PathLike] = None,
    show_tok=True,
    show_doc=True,
):
    """Parse a directory of raw text documents, one by one.

    Args:
        parser (SpacyBISTParser)
        input_dir (str or PathLike)
        out_dir (str or PathLike): If specified, the output will also be written to this path.
        show_tok (bool, optional): Specifies whether to include token text in output.
        show_doc (bool, optional): Specifies whether to include document text in output.

    Yields:
        CoreNLPDoc: the annotated document.
    """
    if out_dir:
        print("Writing parsed documents to {}".format(out_dir))
    for filename, file_contents in tqdm(list(_walk_directory(input_dir)), file=sys.stdout):
        parsed_doc = parser.parse(file_contents, show_tok, show_doc)

        if out_dir:
            out_path = Path(out_dir) / (filename + ".json")
            with open(out_path, "w", encoding="utf-8") as file:
                file.write(parsed_doc.pretty_json())
        yield parsed_doc


def _read_lexicon_from_csv(lexicon_path: Union[str, PathLike]) -> dict:
    """Read a lexicon from a CSV file.

    Returns:
        Dictionary of LexiconElements, each LexiconElement presents a row.
    """
    lexicon = {}
    with open(INFERENCE_LEXICONS / lexicon_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter=",", quotechar="|")
        for row in reader:
            try:
                lexicon[row[0]] = LexiconElement(
                    term=row[0], score=row[1], polarity=None, is_acquired=None, position=row[2]
                )
            except Exception:
                lexicon[row[0]] = LexiconElement(
                    term=row[0], score=row[1], polarity=None, is_acquired=None, position=None
                )
    return lexicon


def load_opinion_lex(file_name: Union[str, PathLike]) -> dict:
    """Read opinion lexicon from CSV file.

    Returns:
        Dictionary of LexiconElements, each LexiconElement presents a row.
    """
    lexicon = {}
    with open(file_name, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        next(reader)
        for row in reader:
            term, score, polarity, is_acquired = row[0], row[1], row[2], row[3]
            score = float(score)
            # ignore terms with low score
            if score >= 0.5 and polarity in (Polarity.POS.value, Polarity.NEG.value):
                lexicon[term] = LexiconElement(
                    term.lower(),
                    score if polarity == Polarity.POS.value else -score,
                    polarity,
                    is_acquired,
                )
    return lexicon


def _load_aspect_lexicon(file_name: Union[str, PathLike]):
    """Read aspect lexicon from CSV file.

    Returns: Dictionary of LexiconElements, each LexiconElement presents a row.
    """
    lexicon = []
    with open(file_name, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file, delimiter=",", quotechar="|")
        next(reader)
        for row in reader:
            lexicon.append(LexiconElement(row))
    return lexicon


def _load_parsed_docs_from_dir(directory: Union[str, PathLike]):
    """Read all file in directory.

    Args:
        directory (PathLike): path
    """
    res = {}
    for file_name in listdir(directory):
        if file_name.endswith(".txt") or file_name.endswith(".json"):
            with open(Path(directory) / file_name, encoding="utf-8") as f:
                content = f.read()
                res[file_name] = json.loads(content, object_hook=CoreNLPDoc.decoder)
    return res


def _write_table(table: list, filename: Union[str, PathLike]):
    """Write table as csv to file system.

    Args:
        table (list): table to be printed, as list of lists
        filename (str or Pathlike): file name
    """
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",")
        for row in table:
            writer.writerow(row)


def _write_final_opinion_lex(dictionary: list, file_name: Union[str, PathLike]):
    """Write generated opinion lex as csv to file.

    Args:
        dictionary (list): list of filtered terms
        file_name (str): file name
    """
    candidate_terms = [["#", "CandidateTerm", "Frequency", "Polarity"]]
    term_num = 1
    for candidate_term in dictionary:
        term_row = [int(term_num)] + candidate_term.as_string_list()
        candidate_terms.append(term_row)
        term_num += 1
    _write_table(candidate_terms, file_name)


def _write_final_aspect_lex(dictionary: list, file_name: Union[str, PathLike]):
    """Write generated aspect lex as csv to file.

    Args:
        dictionary (list): list of filtered terms
        file_name (str or PathLike): file name
    """
    candidate_terms = [["Term"]]
    candidate_terms_debug = [["Frequency", "Term", "Lemma"]]
    for candidate_term in dictionary:
        term_row_debug = candidate_term.as_string_list_aspect_debug()
        term_row = candidate_term.as_string_list_aspect()
        candidate_terms_debug.append(term_row_debug)
        candidate_terms.append(term_row)
    _write_table(candidate_terms, file_name)
    _write_table(candidate_terms_debug, str(path.splitext(file_name)[0]) + "_debug.csv")


def _write_generic_sentiment_terms(dictionary: dict, file_name: Union[str, PathLike]):
    """Write generic sentiment terms as csv to file system."""
    generic_terms = [["Term", "Score", "Polarity"]]
    for generic_term in dictionary.values():
        generic_terms.append([str(generic_term), "1.0", generic_term.polarity.name])
    _write_table(generic_terms, file_name)


def _load_lex_as_list_from_csv(file_name: Union[str, PathLike]):
    """Load lexicon as list.

    Args:
        file_name (str or PathLike): input csv file name
    """
    lexicon_table = []
    with open(file_name, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, skipinitialspace=True)

        if reader is None:
            print("file name is None")
            return lexicon_table

        for row in reader:
            term = row["Term"].strip()
            lexicon_table.append(term)

        return lexicon_table


def read_generic_lex_from_file(file_name: Union[str, PathLike]):
    """Read generic opinion lex for term acquisition.

    Args:
        file_name (str or PathLike): name of csv file
    """
    with open(file_name, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        dict_list = {}
        for row in reader:
            if row["UsedForAcquisition"] == "Y":
                key_term = row["Term"] if len(row) > 0 else ""
                terms = []
                if len(row) > 0:
                    terms = key_term.split()
                polarity = Polarity[row["Polarity"]]
                dict_list[key_term] = OpinionTerm(terms, polarity)
    return dict_list


def _read_generic_lex_for_similarity(file_name: Union[str, PathLike]):
    """Read generic opinion terms for similarity calc from csv file.

    Args:
        file_name (str or PathLike): name of csv file
    """
    with open(file_name, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        dict_list = {}
        for row in reader:
            if row["UsedForReranking"] == "Y":
                key_term = row["Term"] if len(row) > 0 else ""
                polarity = row["Polarity"]
                dict_list[key_term] = polarity
    return dict_list


def _write_aspect_lex(parsed_data: Union[str, PathLike], generated_aspect_lex: dict, out_dir: Path):
    parsed_docs = _load_parsed_docs_from_dir(parsed_data)
    aspect_dict = {}
    max_examples = 20
    label = "AS"
    for doc in parsed_docs.values():
        for sent_text, _ in doc.sent_iter():

            for term, lemma in generated_aspect_lex.items():
                if term in sent_text.lower():
                    _find_aspect_in_sentence(
                        term, lemma, sent_text, aspect_dict, label, max_examples, False
                    )
                if lemma != "" and lemma in sent_text.lower():
                    _find_aspect_in_sentence(
                        term, lemma, sent_text, aspect_dict, label, max_examples, True
                    )

    # write aspect lex to file
    header_row = ["Term", "Alias1", "Alias2", "Alias3"]
    for k in range(1, max_examples + 1):
        header_row.append("Example" + str(k))
    aspect_table = [header_row]

    for [term, lemma], sentences in aspect_dict.items():
        term_row = [term, lemma, "", ""]
        for sent in sentences:
            term_row.append(sent)
        aspect_table.append(term_row)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file_path = out_dir / "generated_aspect_lex.csv"
    _write_table(aspect_table, out_file_path)
    print("Aspect lexicon written to {}".format(out_file_path))


def _find_aspect_in_sentence(term, lemma, sent_text, aspect_dict, label, max_examples, found_lemma):
    search_term = term
    if found_lemma:
        search_term = lemma

    start_idx = sent_text.lower().find(search_term)
    end_idx = start_idx + len(search_term)
    if (start_idx - 1 > 0 and sent_text[start_idx - 1] == " ") and (
        end_idx < len(sent_text) and sent_text[end_idx] == " "
    ):

        sent_text_html = "".join(
            (
                sent_text[:start_idx],
                '<span class="',
                label,
                '">',
                sent_text[start_idx:end_idx],
                "</span>",
                sent_text[end_idx:],
            )
        )

        if (term, lemma) in aspect_dict:
            if len(aspect_dict[term, lemma]) < max_examples:
                aspect_dict[term, lemma].append(str(sent_text_html))
        else:
            aspect_dict[term, lemma] = [sent_text_html]


def _write_opinion_lex(parsed_data, generated_opinion_lex_reranked, out_dir):

    parsed_docs = _load_parsed_docs_from_dir(parsed_data)
    opinion_dict = {}
    max_examples = 20
    label = "OP"
    for doc in parsed_docs.values():
        for sent_text, _ in doc.sent_iter():

            for term, terms_params in generated_opinion_lex_reranked.items():
                is_acquired = terms_params[2]
                if is_acquired == "Y":
                    if term in sent_text.lower():
                        _find_opinion_in_sentence(
                            term, terms_params, sent_text, opinion_dict, label, max_examples
                        )
                else:
                    opinion_dict[term] = list(terms_params)

    # write opinion lex to file
    header_row = ["Term", "Score", "Polarity", "isAcquired"]
    for k in range(1, max_examples + 1):
        header_row.append("Example" + str(k))
    opinion_table = [header_row]

    for term, value in opinion_dict.items():
        term_row = [term, value[0], value[1], value[2]]
        for k in range(3, len(value)):
            term_row.append(value[k])
        opinion_table.append(term_row)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file_path = out_dir / "generated_opinion_lex_reranked.csv"
    _write_table(opinion_table, out_file_path)
    print("Reranked opinion lexicon written to {}".format(out_file_path))


def _find_opinion_in_sentence(term, terms_params, sent_text, opinion_dict, label, max_examples):

    start_idx = sent_text.lower().find(term)
    end_idx = start_idx + len(term)

    if (start_idx - 1 > 0 and sent_text[start_idx - 1] == " ") and (
        end_idx < len(sent_text) and sent_text[end_idx] == " "
    ):

        sent_text_html = "".join(
            (
                sent_text[:start_idx],
                '<span class="',
                label,
                '">',
                sent_text[start_idx:end_idx],
                "</span>",
                sent_text[end_idx:],
            )
        )

        if term in opinion_dict:
            if len(opinion_dict[term]) < max_examples:
                opinion_dict[term].append(str(sent_text_html))
        else:
            vals = list(terms_params)
            vals.append(str(sent_text_html))
            opinion_dict[term] = vals
