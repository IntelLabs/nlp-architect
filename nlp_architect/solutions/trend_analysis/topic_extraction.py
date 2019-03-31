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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import csv
import os
import logging
import sys
from multiprocessing import Pool
from os import path, makedirs

from fastText import train_unsupervised
from newspaper import Article

from nlp_architect.solutions.trend_analysis.np_scorer import NPScorer
from nlp_architect import LIBRARY_OUT
from nlp_architect.utils.io import validate_existing_directory
from nlp_architect.utils.text import SpacyInstance

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
data_dir = str(LIBRARY_OUT / 'trend-analysis-data')


def noun_phrase_extraction(docs, parser):
    """
    Extract noun-phrases from a textual corpus

    Args:
        docs (List[String]): A list of documents
        parser (SpacyInstance): Spacy NLP parser
    Returns:
        List of topics with their tf_idf, c_value, language-model scores
    """
    logger.info('extracting NPs')
    np_app = NPScorer(parser=parser)
    np_result = np_app.score_documents(docs, return_all=True)
    logger.info('got NP extractor result. %s phrases were extracted', str(len(np_result)))
    return np_result


def load_text_from_folder(folder):
    """
    Load files content into a list of docs (texts)

    Args:
        folder: A path to a folder containing text files

    Returns:
        A list of documents (List[String])
    """
    files_content = []
    try:
        for filename in os.listdir(folder):
            with open(folder + os.sep + filename, "rb") as f:
                try:
                    files_content.append(
                        ((f.read()).decode('UTF-8', errors='replace')).replace(
                            '\n', ' ').replace('\r',
                                               ' ').replace(
                            '\b', ' '))
                except Exception as e:
                    logger.error(str(e))
    except Exception as e:
        logger.error("Error in load_text: %s", str(e))

    return files_content


def load_url_content(url_list):
    """
    Load articles content into a list of docs (texts)

    Args:
        url_list (List[String]): A list of urls

    Returns:
        A list of documents (List[String])
    """
    files_content = []
    url = ''
    try:
        for url in url_list:
            try:
                url = str(url)
                logger.info("loading %s", url)
                article = Article(url)
                article.download()
                article.parse()
                files_content.append(article.title + ' ' + article.text)
            except Exception as e:
                logger.error(str(e))
    except Exception as e:
        logger.error("Error in load_text: %s, for url: %s", str(e), str(url))

    return files_content


def save_scores(np_result, file_path):
    """
    Save the result of a topic extraction into a file

    Args:
        np_result: A list of topics with different score types (tfidf, cvalue, freq)
        file_path: The output file path
    """
    logger.info('saving multi-scores np extraction results to: %s', file_path)
    with open(file_path, 'wt', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        ctr = 0
        for group, tfidf, cvalue, freq in np_result:
            try:
                group_str = ""
                for p in group:
                    group_str += p + ';'
                row = (group_str[0:-1], str(tfidf), str(cvalue), str(freq))
                writer.writerow(row)
                ctr += 1
            except Exception as e:
                logger.error("Error while writing np results. iteration #: %s. Error: %s", str(
                    ctr), str(e))


def unify_corpora_from_texts(text_list_t, text_list_r):
    """
    Merge two corpora into a single text file

    Args:
        text_list_t: A list of documents - target corpus (List[String])
        text_list_r: A list of documents - reference corpus (List[String])
    Returns:
        The path of the unified corpus
    """
    logger.info('prepare data for w2v training')
    out_file = str(path.join(data_dir, 'corpus.txt'))
    try:
        with open(out_file, 'w+', encoding='utf-8') as writer:
            write_text_list_to_file(text_list_t, writer)
            write_text_list_to_file(text_list_r, writer)
    except Exception as e:
        logger.error('Error: %s', str(e))
    return out_file


def unify_corpora_from_folders(corpus_a, corpus_b):
    """
    Merge two corpora into a single text file

    Args:
        corpus_a: A folder containing text files (String)
        corpus_b: A folder containing text files (String)
    Returns:
        The path of the unified corpus
    """
    logger.info('prepare data for w2v training')
    out_file = str(path.join(data_dir, 'corpus.txt'))
    try:
        with open(out_file, 'w+', encoding='utf-8') as writer:
            write_folder_corpus_to_file(corpus_a, writer)
            write_folder_corpus_to_file(corpus_b, writer)
    except Exception as e:
        logger.error('Error: %s', str(e))
    return out_file


def write_folder_corpus_to_file(corpus, writer):
    """
    Merge content of a folder into a single text file

    Args:
        corpus: A folder containing text files (String)
        writer: A file writer
    """
    for filename in os.listdir(corpus):
        try:
            file_path = str(path.join(corpus, filename))
            with open(file_path, "r", encoding='utf-8') as f:
                lines = f.readlines()
                writer.writelines(lines)
        except Exception as e:
            logger.error('Error: %s. skipping file: %s', str(e), str(filename))


def write_text_list_to_file(text_list, writer):
    """
    Merge a list of texts into a single text file

    Args:
        text_list: A list of texts (List[String])
        writer: A file writer
    """
    try:
        for text in text_list:
            writer.writelines(text)
    except Exception as e:
        logger.error('Error: %s', str(e))


def train_w2v_model(data):
    """
    Train a w2v (skipgram) model using fasttext package

    Args:
        data: A path to the training data (String)
    """
    logger.info('Fasttext embeddings training...')
    try:
        model = train_unsupervised(input=data, model='skipgram', epoch=100, minCount=1, dim=100)
        model.save_model(str(path.join(data_dir, 'W2V_Models/model.bin')))
    except Exception as e:
        logger.error('Error: %s', str(e))


def create_w2v_model(text_list_t, text_list_r):
    """
     Create a w2v model on the given corpora

     Args:
         text_list_t: A list of documents - target corpus (List[String])
         text_list_r: A list of documents - reference corpus (List[String])
     """
    unified_data = unify_corpora_from_texts(text_list_t, text_list_r)
    train_w2v_model(unified_data)


def get_urls_from_file(file):
    """
     Merge two corpora into a single text file

     Args:
         corpus_a: A folder containing text files (String)
         corpus_b: A folder containing text files (String)
     Returns:
         The path of the unified corpus
     """
    with open(file) as url_file:
        url_list = url_file.readlines()
        url_list = [x.strip() for x in url_list]
    return url_list


def initiate_parser():
    return SpacyInstance(
        disable=['tagger', 'ner', 'parser', 'vectors', 'textcat']).parser


def main(corpus_t, corpus_r, single_thread, no_train, url):
    try:
        if not path.exists(data_dir):
            makedirs(data_dir)
        if not path.exists(path.join(data_dir, 'W2V_Models')):
            os.makedirs(path.join(data_dir, 'W2V_Models'))
    except Exception:
        logger.error('failed to create output folder.')
        sys.exit()
    if url:
        if not path.isfile(corpus_t) or not path.isfile(corpus_r):
            logger.error('Please provide a valid csv file with urls')
            sys.exit()
        else:
            text_t = load_url_content(get_urls_from_file(corpus_t))
            text_r = load_url_content(get_urls_from_file(corpus_r))
    else:
        if not path.isdir(corpus_t) or not path.isdir(corpus_r):
            logger.error('Please provide valid directories for target_corpus and'
                         ' for ref_corpus')
            sys.exit()

        else:
            text_t = load_text_from_folder(corpus_t)
            text_r = load_text_from_folder(corpus_r)

    # extract noun phrases
    nlp_parser_t = initiate_parser()
    nlp_parser_r = initiate_parser()
    if single_thread:
        result_t = noun_phrase_extraction(text_t, nlp_parser_t)
        result_r = noun_phrase_extraction(text_r, nlp_parser_r)
    else:
        with Pool(processes=2) as pool:
            run_np_t = pool.apply_async(noun_phrase_extraction,
                                        [text_t, nlp_parser_t])
            run_np_r = pool.apply_async(noun_phrase_extraction,
                                        [text_r, nlp_parser_r])
            result_t = run_np_t.get()
            result_r = run_np_r.get()

    # save results to csv

    # save_scores(result_t, os.path.splitext(os.path.basename(corpus_t))[0] + '.csv')
    # save_scores(result_r, os.path.splitext(os.path.basename(corpus_r))[0] + '.csv')
    save_scores(result_t, str(path.join(data_dir, os.path.basename(corpus_t))) + '.csv')
    save_scores(result_r, str(path.join(data_dir, os.path.basename(corpus_r))) + '.csv')

    # create w2v model
    if not no_train:
        create_w2v_model(text_t, text_r)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(prog='topic_extraction.py')
    argparser.add_argument('target_corpus', metavar='target_corpus',
                           type=validate_existing_directory,
                           help='a path to a folder containing text files')
    argparser.add_argument('ref_corpus', metavar='ref_corpus', type=validate_existing_directory,
                           help='a path to a folder containing text files')
    argparser.add_argument('--no_train', action='store_true',
                           help='skip the creation of w2v model')
    argparser.add_argument('--url', action='store_true',
                           help='corpus provided as csv file with urls')
    argparser.add_argument('--single_thread', action='store_true',
                           help='analyse corpora sequentially')
    args = argparser.parse_args()

    main(args.target_corpus, args.ref_corpus, args.single_thread, args.no_train, args.url)
