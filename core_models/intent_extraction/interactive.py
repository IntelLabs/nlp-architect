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

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os

import argparse
import sys
from builtins import input

import numpy as np
from nlp_architect.data.intent_datasets import SNIPS
from nlp_architect.models.intent_extraction import IntentExtractionModel
from nlp_architect.utils.embedding import load_word_embeddings
from nlp_architect.utils.text import SpacyTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True,
                    help='Model file path')
parser.add_argument('--dataset_path', type=str, required=True,
                    help='dataset directory')
parser.add_argument('--embedding_model', type=str,
                    help='Path to word embedding model')
parser.add_argument('--embedding_size', type=int,
                    help='Word embedding model vector size')
args = parser.parse_args()

if not os.path.exists(args.model_path):
    print('model_path does not exist')
    sys.exit(0)
if not os.path.exists(args.dataset_path):
    print('dataset_path does not exist')
    sys.exit(0)
if args.embedding_model is not None and not os.path.exists(args.embedding_model):
    print('word embedding model file was not found')
    sys.exit(0)

model = IntentExtractionModel()
model.load(args.model_path)

ds = SNIPS(path=args.dataset_path)
nlp = SpacyTokenizer()


def process_text(text):
    max_sen_size = model.input_shape[1]
    tokens = nlp.tokenize(u'{}'.format(text))
    return [t for t in tokens[:max_sen_size] if len(t.strip()) > 0]


def encode_sentence(tokens, emb_vectors=None):
    max_sen_size = model.input_shape[1]
    if emb_vectors is not None:
        input_vec = np.zeros((max_sen_size, args.embedding_size))
        zeros = np.zeros(args.embedding_size)
        tvecs = [emb_vectors.get(t.lower(), zeros) for t in tokens]
        input_vec[-len(tvecs):] = tvecs
        input_vec = input_vec.reshape((1, max_sen_size, -1))
    else:
        input_vec = np.zeros((max_sen_size,))
        tids = [ds.tokens_vocab.get(t.lower(), 1) for t in tokens]
        input_vec[-len(tids):] = tids
        input_vec = input_vec.reshape((1, -1))
    return input_vec


def display_results(tokens, predictions):
    if type(predictions) == list:
        tags = predictions[1][0].argmax(1)
        intent = predictions[0].argmax()
        iv = {v: k for k, v in ds.intents_vocab.items()}
        print('Intent type: {}'.format(iv.get(intent)))
    else:
        tags = predictions[0].argmax(1)
    sv = {v: k for k, v in ds.labels_vocab.items()}
    print_helper = []
    for t, p in zip(tokens, tags[-len(tokens):]):
        tag = sv.get(p, 'OOV')
        print_helper.append((t, tag, max(len(t), len(tag))))
    print(' '.join(['%-{}s'.format(x[2]) % x[0] for x in print_helper]))
    print(' '.join(['%-{}s'.format(x[2]) % '|' for x in print_helper]))
    print(' '.join(['%-{}s'.format(x[2]) % x[1] for x in print_helper]))


emb_vectors = None
if args.embedding_model is not None and args.embedding_size is not None:
    print('Loading external word embedding model of size {}'.format(args.embedding_size))
    emb_vectors, _ = load_word_embeddings(args.embedding_model)

while True:
    text = input('>> ')
    tokens = process_text(text)
    enc_sent = encode_sentence(tokens, emb_vectors)
    predictions = model.predict(enc_sent)
    display_results(tokens, predictions)
