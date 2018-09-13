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

"""
Script that prepares the input corpus for np2vec training: it runs NP extractor on the corpus and
marks extracted NP's.
"""
import gzip
import logging
import sys
import json
from os import path
from configargparse import ArgumentParser

from nlp_architect.pipelines.spacy_np_annotator import NPAnnotator, get_noun_phrases
from nlp_architect.utils.text import spacy_normalizer, SpacyInstance
from nlp_architect.utils.io import check_size, validate_existing_filepath, \
    download_unlicensed_file
from tqdm import tqdm

chunker_model_dat_file = 'model_info.dat.params'
chunker_model_file = 'model.h5'

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

np2id = {}
id2group = {}
id2rep = {}
np2count = {}
cur_dir = path.dirname(path.realpath(__file__))
nlp_chunker_url = 'http://nervana-modelzoo.s3.amazonaws.com/NLP/chunker/'


def get_group_norm(spacy_span):
    """
    Give a span, determine the its group and return the normalized text representing the group

    Args:
            spacy_span (spacy.tokens.Span)
    """
    np = spacy_span.text
    if np not in np2count:
        np2count[np] = 1
    else:
        np2count[np] += 1
    norm = spacy_normalizer(np, spacy_span.lemma_)
    if args.mark_char in norm:
        norm = norm.replace(args.mark_char, ' ')
    np2id[np] = norm
    if norm not in id2rep:
        id2rep[norm] = np
    if norm in id2group:
        if np not in id2group[norm]:
            id2group[norm].append(np)
        elif np2count[np] > np2count[id2rep[norm]]:
            id2rep[norm] = np  # replace rep
    else:
        id2group[norm] = [np]
        id2rep[norm] = np
    return norm


if __name__ == '__main__':
    arg_parser = ArgumentParser(__doc__)
    arg_parser.add_argument(
        '--corpus',
        default=path.abspath(
            cur_dir + "../../../datasets/wikipedia/enwiki-20171201_subset.txt.gz"),
        type=validate_existing_filepath,
        help='path to the input corpus. Compressed files (gz) are also supported. By default, '
             'it is a subset of English Wikipedia.')
    arg_parser.add_argument(
        '--marked_corpus',
        default='enwiki-20171201_subset_marked.txt',
        type=str,
        action=check_size(min_size=1),
        help='path to the marked corpus corpus.')
    arg_parser.add_argument(
        '--mark_char',
        default='_',
        type=str,
        action=check_size(1, 2),
        help='special character that marks NP\'s in the corpus (word separator and NP suffix). '
             'Default value is _.')
    arg_parser.add_argument(
        '--grouping',
        action='store_true',
        default=False,
        help='perform noun-phrase grouping')
    arg_parser.add_argument(
        '--chunker', type=str,
        choices=['spacy', 'nlp_arch'],
        default='spacy',
        help='chunker to use for detecting noun phrases. \'spacy\' for using spacy built-in '
             'chunker or \'nlp_arch\' for NLP Architect NP Extractor')

    args = arg_parser.parse_args()
    if args.corpus.endswith('gz'):
        corpus_file = gzip.open(args.corpus, 'rt', encoding='utf8', errors='ignore')
    else:
        corpus_file = open(args.corpus, 'r', encoding='utf8', errors='ignore')

    with open(args.marked_corpus, 'w', encoding='utf8') as marked_corpus_file:
        # load spacy parser
        logger.info('loading spacy')
        if 'nlp_arch' in args.chunker:
            nlp = SpacyInstance(model='en_core_web_sm',
                                disable=['textcat', 'ner', 'parser']).parser
            nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
            logger.info(
                'The pre-trained model to be downloaded for NLP Architect word'
                ' chunker model is licensed under Apache 2.0')
            _path_to_model = path.join(cur_dir, chunker_model_file)
            if not path.exists(_path_to_model):
                download_unlicensed_file(nlp_chunker_url, chunker_model_file, _path_to_model)
            _path_to_params = path.join(cur_dir, chunker_model_dat_file)
            if not path.exists(_path_to_params):
                download_unlicensed_file(nlp_chunker_url, chunker_model_dat_file, _path_to_params)
            logger.info('Done.')
            nlp.add_pipe(NPAnnotator.load(_path_to_model, _path_to_params), last=True)
        else:
            nlp = SpacyInstance(model='en_core_web_sm', disable=['textcat', 'ner']).parser
        logger.info('spacy loaded')

        num_lines = sum(1 for line in corpus_file)
        corpus_file.seek(0)
        logger.info('%i lines in corpus', num_lines)
        i = 0

        with tqdm(total=num_lines) as pbar:
            for doc in nlp.pipe(corpus_file, n_threads=-1):
                if 'nlp_arch' in args.chunker:
                    spans = get_noun_phrases(doc)
                else:
                    spans = list(doc.noun_chunks)
                i += 1
                if len(spans) > 0:
                    span = spans.pop(0)
                else:
                    span = None
                spanWritten = False
                for token in doc:
                    if span is None:
                        if len(token.text.strip()) > 0:
                            marked_corpus_file.write(token.text + ' ')
                    else:
                        if token.idx < span.start_char or token.idx >= span.end_char:  # outside a
                            # span
                            if len(token.text.strip()) > 0:
                                marked_corpus_file.write(token.text + ' ')
                        else:
                            if not spanWritten:
                                # mark NP's
                                if len(span.text) > 1 and span.lemma_ != '-PRON-':
                                    if args.grouping:
                                        text = get_group_norm(span)
                                    else:
                                        text = span.text
                                    # mark NP's
                                    text = text.replace(' ', args.mark_char) + args.mark_char
                                    marked_corpus_file.write(text + ' ')
                                else:
                                    marked_corpus_file.write(span.text + ' ')
                                spanWritten = True
                            if token.idx + len(token.text) == span.end_char:
                                if len(spans) > 0:
                                    span = spans.pop(0)
                                else:
                                    span = None
                                spanWritten = False
                marked_corpus_file.write('\n')
                pbar.update(1)

    # write grouping data :
    if args.grouping:
        corpus_name = path.basename(args.corpus)
        with open(path.join(cur_dir, 'id2group'), 'w', encoding='utf8') as id2group_file:
            id2group_file.write(json.dumps(id2group))

        with open(path.join(cur_dir, 'id2rep'), 'w', encoding='utf8') as id2rep_file:
            id2rep_file.write(json.dumps(id2rep))

        with open(path.join(cur_dir, 'np2id'), 'w', encoding='utf8') as np2id_file:
            np2id_file.write(json.dumps(np2id))

    corpus_file.close()
