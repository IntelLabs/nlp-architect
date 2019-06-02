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
import json
import logging
import sys
from argparse import ArgumentParser
from os import path, makedirs

from tqdm import tqdm

from nlp_architect import LIBRARY_OUT
from nlp_architect.pipelines.spacy_np_annotator import NPAnnotator, get_noun_phrases
from nlp_architect.utils.io import check_size, download_unlicensed_file, validate_parent_exists
from nlp_architect.utils.text import spacy_normalizer, SpacyInstance

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

np2id = {}
id2group = {}
id2rep = {}
np2count = {}
nlp_chunker_url = 'https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/chunker/'
chunker_path = str(LIBRARY_OUT / 'chunker-pretrained')
chunker_model_dat_file = 'model_info.dat.params'
chunker_model_file = 'model.h5'


def get_group_norm(spacy_span):
    """
    Give a span, determine the its group and return the normalized text representing the group

    Args:
            spacy_span (spacy.tokens.Span)
    """
    np = spacy_span.text
    norm = spacy_normalizer(np, spacy_span.lemma_)
    if args.mark_char in norm:
        norm = norm.replace(args.mark_char, ' ')
    if np not in np2count:  # new np
        np2count[np] = 1
        np2id[np] = norm
        if norm in id2group:  # norm already exist
            id2group[norm].append(np)
        else:
            id2group[norm] = [np]
            id2rep[norm] = np
    else:  # another occurrence of this np. norm must exist and be consistent
        np2count[np] += 1
        if np2id[np] != norm:  # new norm to the same np - merge groups.
            #  no need to update np2id[np]
            norm = merge_groups(np, np2id[np], norm)  # set to the already exist
            #  norm so I know this norm is already in id2group/id2rep
        else:  # update rep
            if np2count[np] > np2count[id2rep[norm]]:
                id2rep[norm] = np  # replace rep

    return norm


def load_parser(chunker):
    # load spacy parser
    logger.info('loading spacy. chunker=%s', chunker)
    if 'nlp_arch' in chunker:
        parser = SpacyInstance(model='en_core_web_sm',
                               disable=['textcat', 'ner', 'parser']).parser
        parser.add_pipe(parser.create_pipe('sentencizer'), first=True)
        _path_to_model = path.join(chunker_path, chunker_model_file)
        _path_to_params = path.join(chunker_path, chunker_model_dat_file)
        if not path.exists(chunker_path):
            makedirs(chunker_path)
        if not path.exists(_path_to_model):
            logger.info(
                'The pre-trained model to be downloaded for NLP Architect'
                ' word chunker model is licensed under Apache 2.0')
            download_unlicensed_file(nlp_chunker_url, chunker_model_file, _path_to_model)
        if not path.exists(_path_to_params):
            download_unlicensed_file(nlp_chunker_url, chunker_model_dat_file, _path_to_params)
        parser.add_pipe(NPAnnotator.load(_path_to_model, _path_to_params),
                        last=True)
    else:
        parser = SpacyInstance(model='en_core_web_sm', disable=['textcat', 'ner']).parser
    logger.info('spacy loaded')
    return parser


def extract_noun_phrases(docs, nlp_parser, chunker):
    logger.info('extract nps from: %s', docs)
    spans = []
    for doc in nlp_parser.pipe(docs, n_threads=-1):
        if 'nlp_arch' in chunker:
            spans.extend(get_noun_phrases(doc))
        else:
            nps = list(doc.noun_chunks)
            spans.extend(nps)
    logger.info('nps= %s', str(spans))
    return spans


# pylint: disable-msg=too-many-nested-blocks,too-many-branches
def mark_noun_phrases(corpus_file, marked_corpus_file, nlp_parser, lines_count, chunker,
                      mark_char='_', grouping=False):
    i = 0
    with tqdm(total=lines_count) as pbar:
        for doc in nlp_parser.pipe(corpus_file, n_threads=-1):
            if 'nlp_arch' in chunker:
                spans = get_noun_phrases(doc)
            else:
                spans = list(doc.noun_chunks)
            i += 1
            if len(spans) > 0:
                span = spans.pop(0)
            else:
                span = None
            span_written = False
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
                        if not span_written:
                            # mark NP's
                            if len(span.text) > 1 and span.lemma_ != '-PRON-':
                                if grouping:
                                    text = get_group_norm(span)
                                else:
                                    text = span.text
                                # mark NP's
                                text = text.replace(' ',
                                                    mark_char) + mark_char
                                marked_corpus_file.write(text + ' ')
                            else:
                                marked_corpus_file.write(span.text + ' ')
                            span_written = True
                        if token.idx + len(token.text) == span.end_char:
                            if len(spans) > 0:
                                span = spans.pop(0)
                            else:
                                span = None
                            span_written = False
            marked_corpus_file.write('\n')
            pbar.update(1)


def merge_groups(np, old_id, diff_id):
    if diff_id in id2group:
        for term in id2group[diff_id]:  # for each term update dicts
            np2id[term] = old_id
            id2group[old_id].append(term)
            if np2count[term] > np2count[id2rep[old_id]]:
                id2rep[old_id] = term
        id2rep.pop(diff_id)
        id2group.pop(diff_id)
    else:
        if np2count[np] > np2count[id2rep[old_id]]:
            id2rep[old_id] = np
    return old_id


if __name__ == '__main__':
    arg_parser = ArgumentParser(__doc__)
    arg_parser.add_argument(
        '--corpus',
        help='path to the input corpus. Compressed files (gz) are also supported. By default, '
             'it is a subset of English Wikipedia. '
             'get subset of English wikipedia from '
             'https://github.com/NervanaSystems/nlp-architect/raw/'
             'master/datasets/wikipedia/enwiki-20171201_subset.txt.gz')
    arg_parser.add_argument(
        '--marked_corpus',
        default='enwiki-20171201_subset_marked.txt',
        type=validate_parent_exists,
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
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'

    with open_func(args.corpus, mode, encoding='utf8', errors='ignore') as my_corpus_file:
        with open(args.marked_corpus, 'w', encoding='utf8') as my_marked_corpus_file:
            nlp = load_parser(args.chunker)
            num_lines = sum(1 for line in my_corpus_file)
            my_corpus_file.seek(0)
            logger.info('%i lines in corpus', num_lines)
            mark_noun_phrases(my_corpus_file, my_marked_corpus_file, nlp, num_lines,
                              mark_char=args.mark_char, grouping=args.grouping,
                              chunker=args.chunker)

        # write grouping data :
        if args.grouping:
            corpus_dir = path.dirname(args.marked_corpus)
            with open(path.join(corpus_dir, 'id2group'), 'w', encoding='utf8') as id2group_file:
                id2group_file.write(json.dumps(id2group))

            with open(path.join(corpus_dir, 'id2rep'), 'w', encoding='utf8') as id2rep_file:
                id2rep_file.write(json.dumps(id2rep))

            with open(path.join(corpus_dir, 'np2id'), 'w', encoding='utf8') as np2id_file:
                np2id_file.write(json.dumps(np2id))
