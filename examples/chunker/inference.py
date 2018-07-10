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

from __future__ import division, print_function, unicode_literals, absolute_import

import argparse
import pickle
from os import path

import numpy as np

from nlp_architect.models.chunker import SequenceChunker
from nlp_architect.utils.io import validate_existing_filepath
from nlp_architect.utils.text import SpacyInstance


def vectorize(docs, vocab):
    data = []
    for doc in docs:
        data.append(np.asarray([vocab[w] if vocab[w] is not None else 1 for w in doc])
                    .reshape(1, -1))
    return data


def build_annotation(documents, annotations):
    for d, i in zip(documents, annotations):
        for w, p, c in zip(d, i[0], i[1]):
            print('{}\t{}\t{}'.format(w, p, c))
        print('')


if __name__ == '__main__':
    # read input args and validate
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=validate_existing_filepath, required=True,
                        help='Input texts file path (samples to pass for inference)')
    parser.add_argument('--model_name', default='chunker_model', type=str,
                        required=True, help='Model name (used for saving the model)')
    args = parser.parse_args()
    model_path = path.join(path.dirname(path.realpath(__file__)),
                           '{}.h5'.format(str(args.model_name)))
    settings_path = path.join(path.dirname(path.realpath(__file__)),
                              '{}.params'.format(str(args.model_name)))
    validate_existing_filepath(model_path)
    validate_existing_filepath(settings_path)

    # load model and parameters
    model = SequenceChunker()
    model.load(model_path)
    with open(settings_path, 'rb') as fp:
        model_params = pickle.load(fp)
        word_vocab = model_params['word_vocab']
        chunk_vocab = model_params['chunk_vocab']
        pos_vocab = model_params['pos_vocab']

    # parse documents and get tokens
    nlp = SpacyInstance(disable=['tagger', 'ner', 'parser', 'vectors', 'textcat'])
    with open(args.input_file) as fp:
        documents = [nlp.tokenize(t.strip()) for t in fp.readlines()]

    # vectorize input tokens and run inference
    doc_vecs = vectorize(documents, word_vocab)
    annotations = []
    for doc in doc_vecs:
        doc_pos, doc_chunks = model.predict(doc, batch_size=1)
        pos_a = [pos_vocab.id_to_word(l) for l in doc_pos.argmax(2).flatten()]
        chunk_a = [chunk_vocab.id_to_word(l) for l in doc_chunks.argmax(2).flatten()]
        annotations.append((pos_a, chunk_a))

    # print document text and annotations
    build_annotation(documents, annotations)
