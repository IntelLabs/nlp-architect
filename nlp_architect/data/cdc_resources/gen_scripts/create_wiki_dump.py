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
import argparse
import json
import logging

from nlp_architect.data.cdc_resources.relations.relation_types_enums import WikipediaSearchMethod
from nlp_architect.data.cdc_resources.relations.wikipedia_relation_extraction import \
    WikipediaRelationExtraction
from nlp_architect.models.cross_doc_coref.system.cdc_utils import load_mentions_vocab_from_files
from nlp_architect.utils import io
from nlp_architect.utils.io import json_dumper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

result_dump = {}

parser = argparse.ArgumentParser(description='Create Wikipedia dataset only dump')
parser.add_argument('--mentions', type=str, help='mentions_file file', required=True)
parser.add_argument('--host', type=str, help='elastic host')
parser.add_argument('--port', type=int, help='elastic port')
parser.add_argument('--index', type=str, help='elastic index')
parser.add_argument('--output', type=str, help='location were to create dump file', required=True)

args = parser.parse_args()


def wiki_dump_from_gs():
    logger.info('Starting, process will connect with ElasticSearch and online wikipedia site...')
    mentions_files = [args.mentions]
    dump_file = args.output
    vocab = load_mentions_vocab_from_files(mentions_files)

    if args.host and args.port and args.index:
        wiki_elastic = WikipediaRelationExtraction(WikipediaSearchMethod.ELASTIC,
                                                   host=args.host,
                                                   port=args.port,
                                                   index=args.index)
    else:
        logger.info(
            'Running without Wikipedia elastic search, Note that this will '
            'take much longer to process only using online service')
        wiki_elastic = None

    wiki_online = WikipediaRelationExtraction(WikipediaSearchMethod.ONLINE)

    for phrase in vocab:
        phrase = phrase.replace("'", "").replace('"', "").replace('\\', "").strip()
        logger.info('Try to retrieve \'%s\' from elastic search', phrase)
        pages = None
        if wiki_elastic:
            pages = wiki_elastic.get_phrase_related_pages(phrase)
        if not pages or not pages.get_pages() or len(pages.get_pages()) == 0:
            logger.info('Not on elastic, retrieve \'%s\' from wiki online site', phrase)
            pages = wiki_online.get_phrase_related_pages(phrase)
        for search_page in pages.get_pages():
            add_page(search_page, phrase)

    with open(dump_file, 'w') as myfile:
        json.dump(result_dump, myfile, default=json_dumper)

    logger.info('Saving dump to file-%s', dump_file)


def add_page(search_page, phrase):
    try:
        if search_page is not None:
            if phrase not in result_dump:
                result_dump[phrase] = []
                result_dump[phrase].append(search_page)
            else:
                pages = result_dump[phrase]
                for page in pages:
                    if page.pageid == search_page.pageid:
                        return

                result_dump[phrase].append(search_page)

            logger.info('page-%s added', str(search_page))
    except Exception:
        logger.error('could not extract wiki info from phrase-%s', search_page.orig_phrase)


if __name__ == '__main__':
    io.validate_existing_filepath(args.mentions)
    io.validate_existing_filepath(args.output)
    if args.host:
        io.validate((args.host, str, 1, 1000))
    if args.port:
        io.validate((args.port, int, 1, 65536))
    if args.index:
        io.validate((args.index, str, 1, 10000))

    wiki_dump_from_gs()
