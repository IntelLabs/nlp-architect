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
import spacy
from configargparse import ArgumentParser
from nlp_architect.utils.io import check_size, validate_existing_filepath

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    arg_parser = ArgumentParser(__doc__)
    arg_parser.add_argument(
        '--corpus',
        default='../../datasets/wikipedia/enwiki-20171201_subset.txt.gz',
        type=validate_existing_filepath,
        action=check_size(min_size=1),
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

    args = arg_parser.parse_args()

    if args.corpus.endswith('gz'):
        corpus_file = gzip.open(args.corpus, 'rt', encoding='utf8')
    else:
        corpus_file = open(args.corpus, 'r', encoding='utf8')

    with open(args.marked_corpus, 'w', encoding='utf8') as marked_corpus_file:

        # spacy NP extractor
        logger.info('loading spacy')
        nlp = spacy.load('en_core_web_sm', disable=['textcat', 'ner'])
        logger.info('spacy loaded')

        num_lines = sum(1 for line in corpus_file)
        corpus_file.seek(0)
        logger.info('%i lines in corpus', num_lines)
        i = 0

        for doc in nlp.pipe(corpus_file):
            spans = list()
            for span in doc.noun_chunks:
                spans.append(span)
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
                                text = span.text.replace(' ', args.mark_char) + args.mark_char
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
            if i % 500 == 0:
                logger.info('%i of %i lines', i, num_lines)

    corpus_file.close()
