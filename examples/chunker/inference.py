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

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import pickle
from os import path

import numpy as np

from nlp_architect.models.chunker import SequenceChunker
from nlp_architect.utils.generic import pad_sentences
from nlp_architect.utils.io import check_size, validate_existing_filepath
from nlp_architect.utils.text import SpacyInstance


def vectorize(docs, w_vocab, c_vocab):
    data = []
    for doc in docs:
        words = np.asarray([w_vocab[w.lower()] if w_vocab[w.lower()] is not None else 1
                            for w in doc]).reshape(1, -1)
        if c_vocab is not None:
            sentence_chars = []
            for w in doc:
                word_chars = []
                for c in w:
                    _cid = c_vocab[c]
                    word_chars.append(_cid if _cid is not None else 1)
                sentence_chars.append(word_chars)
            sentence_chars = np.expand_dims(pad_sentences(sentence_chars, word_length), axis=0)
            data.append((words, sentence_chars))
        else:
            data.append(words)
    return data


def build_annotation(documents, annotations):
    for d, i in zip(documents, annotations):
        for w, c in zip(d, i):
            print('{}\t{}'.format(w, c))
        print('')


if __name__ == '__main__':
    # read input args and validate
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=validate_existing_filepath, required=True,
                        help='Input texts file path (samples to pass for inference)')
    parser.add_argument('--model_name', default='chunker_model', type=str,
                        required=True, help='Model name (used for saving the model)')
    parser.add_argument('-b', type=int, action=check_size(1, 9999), default=1,
                        help='inference batch size')
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
    word_length = model.max_word_len
    with open(settings_path, 'rb') as fp:
        model_params = pickle.load(fp)
        word_vocab = model_params['word_vocab']
        chunk_vocab = model_params['chunk_vocab']
        char_vocab = model_params.get('char_vocab', None)

    # parse documents and get tokens
    nlp = SpacyInstance(disable=['tagger', 'ner', 'parser', 'vectors', 'textcat'])
    with open(args.input_file) as fp:
        document_texts = [nlp.tokenize(t.strip()) for t in fp.readlines()]

    # vectorize input tokens and run inference
    doc_vecs = vectorize(document_texts, word_vocab, char_vocab)
    document_annotations = []
    for vec in doc_vecs:
        doc_chunks = model.predict(vec, batch_size=args.b)
        chunk_a = [chunk_vocab.id_to_word(l) for l in doc_chunks.argmax(2).flatten()]
        document_annotations.append(chunk_a)

    # print document text and annotations
    build_annotation(document_texts, document_annotations)
