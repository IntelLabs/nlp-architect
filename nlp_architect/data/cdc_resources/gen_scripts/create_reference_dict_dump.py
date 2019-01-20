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

from nlp_architect.data.cdc_resources.relations.referent_dict_relation_extraction import \
    ReferentDictRelationExtraction
from nlp_architect.models.cross_doc_coref.system.cdc_utils import load_mentions_vocab_from_files
from nlp_architect.utils import io

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Create Referent dictionary dataset only dump')

parser.add_argument('--ref_dict', type=str, help='referent dictionary file', required=True)

parser.add_argument('--mentions', type=str, help='dataset mentions', required=True)

parser.add_argument('--output', type=str, help='location were to create dump file', required=True)

args = parser.parse_args()


def ref_dict_dump():
    logger.info('Extracting referent dict dump, this may take a while...')
    ref_dict_file = args.ref_dict
    out_file = args.output
    mentions_entity_gold_file = [args.mentions]
    vocab = load_mentions_vocab_from_files(mentions_entity_gold_file, True)

    ref_dict = ReferentDictRelationExtraction.load_reference_dict(ref_dict_file)

    ref_dict_for_vocab = {}
    for word in vocab:
        if word in ref_dict:
            ref_dict_for_vocab[word] = ref_dict[word]

    logger.info('Found %d words from vocabulary', len(ref_dict_for_vocab.keys()))
    logger.info('Preparing to save refDict output file')
    with open(out_file, 'w') as f:
        json.dump(ref_dict_for_vocab, f)
    logger.info('Done saved to-%s', out_file)


if __name__ == '__main__':
    io.validate_existing_filepath(args.mentions)
    io.validate_existing_filepath(args.output)
    io.validate_existing_filepath(args.ref_dict)
    ref_dict_dump()
