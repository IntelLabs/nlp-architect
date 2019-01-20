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
import logging
import os
import pickle
from os.path import join

from nlp_architect.common.cdc.mention_data import MentionData
from nlp_architect.data.cdc_resources.embedding.embed_elmo import ElmoEmbedding
from nlp_architect.utils import io

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_elmo_for_vocab(mentions):
    """
    Create the embedding using the cache logic in the embedding class
    Args:
        mentions:

    Returns:

    """
    elmo_embeddings = ElmoEmbedding()

    for mention in mentions:
        elmo_embeddings.get_head_feature_vector(mention)

    logger.info('Total words/contexts in vocabulary %d', len(elmo_embeddings.cache))
    return elmo_embeddings.cache


def elmo_dump():
    out_file = args.output
    mention_files = list()
    if os.path.isdir(args.mentions):
        for (dirpath, _, files) in os.walk(args.mentions):
            for file in files:
                if file == '.DS_Store':
                    continue

                mention_files.append(join(dirpath, file))
    else:
        mention_files.append(args.mentions)

    mentions = []
    for _file in mention_files:
        mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_file))

    elmo_ecb_embeddings = load_elmo_for_vocab(mentions)

    with open(out_file, 'wb') as f:
        pickle.dump(elmo_ecb_embeddings, f)

    logger.info('Saving dump to file-%s', out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Elmo Embedding dataset only dump')
    parser.add_argument('--mentions', type=str, help='mentions_file file', required=True)
    parser.add_argument('--output', type=str, help='location were to create dump file',
                        required=True)

    args = parser.parse_args()

    if os.path.isdir(args.mentions):
        io.validate_existing_directory(args.mentions)
    else:
        io.validate_existing_filepath(args.mentions)

    elmo_dump()
    print('Done!')
