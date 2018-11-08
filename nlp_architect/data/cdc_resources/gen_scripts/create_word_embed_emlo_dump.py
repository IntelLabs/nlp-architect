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
import pickle

from nlp_architect.data.cdc_resources.embedding.embed_elmo import ElmoEmbedding
from nlp_architect.models.cross_doc_coref.system.cdc_utils import load_mentions_vocab
from nlp_architect.utils import io

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_elmo_for_vocab(vocabulary):
    elmo_embeddings = ElmoEmbedding()
    elmo_dict = dict()
    for mention_string in vocabulary:
        if mention_string not in elmo_dict:
            mention_embedding = elmo_embeddings.get_avrg_feature_vector(mention_string)
            elmo_dict[mention_string] = mention_embedding

    return elmo_dict


def elmo_dump():
    filter_stop_words = False
    out_file = args.output
    mention_files = list()
    mention_files.append(args.mentions)
    vocab = load_mentions_vocab(mention_files, filter_stop_words)
    elmo_ecb_embeddings = load_elmo_for_vocab(vocab)
    logger.info('Total words in vocabulary %d', len(vocab))
    with open(out_file, 'wb') as f:
        pickle.dump(elmo_ecb_embeddings, f)
    logger.info('Saving dump to file-%s', out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Elmo Embedding dataset only dump')
    parser.add_argument('--mentions', type=str, help='mentions_file file', required=True)
    parser.add_argument('--output', type=str, help='location were to create dump file',
                        required=True)
    args = parser.parse_args()
    io.validate_existing_filepath(args.mentions)
    io.validate_existing_filepath(args.output)
    elmo_dump()
