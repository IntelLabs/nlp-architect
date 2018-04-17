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
import pickle
import os

import numpy as np

import spacy
from neon.backends import gen_backend
from neon.models import Model
from neon.util.argparser import NeonArgparser, extract_valid_args

from data import TaggedTextSequence, MultiSequenceDataIterator
from utils import extract_nps, get_word_embeddings, get_paddedXY_sequence

if __name__ == '__main__':

    parser = NeonArgparser()
    parser.add_argument('--model', type=str,
                        help='Path to model file')
    parser.add_argument('--settings', type=str,
                        help='Path to model settings file')
    parser.add_argument('--input', type=str,
                        help='Input texts file path (samples to pass for inference)')
    parser.add_argument('--emb_model', type=str,
                        help='Pre-trained word embedding model file path')
    parser.add_argument('--print_only_nps', default=False, action='store_true',
                        help='Print inferred Noun Phrases')
    args = parser.parse_args()
    be = gen_backend(**extract_valid_args(args, gen_backend))

    model_file = args.model
    if not os.path.exists(args.settings):
        raise Exception('Not valid model settings file')
    else:
        with open(args.settings, 'rb') as fp:
            mp = pickle.load(fp)

    if mp['char_rnn'] is not False:
        raise NotImplementedError

    sentence_len = mp['sentence_len']

    if not os.path.exists(args.input):
        raise Exception('Not valid input file')
    else:
        with open(args.input) as fp:
            input_texts = [t.strip() for t in fp.readlines()]

    be.bsz = len(input_texts)
    input_features = []
    text_tokens = []
    if mp['use_embeddings'] is True:
        emb_model, emb_size = get_word_embeddings(args.emb_model)
        pad_vector = np.zeros(emb_size)
        for t in input_texts:
            vector = [
                emb_model[t] if t in emb_model else pad_vector
                for t in t.lower().split()
            ]
            vector = [pad_vector] * (sentence_len - len(vector)) + vector
            text_tokens.append(np.asarray(vector))
        token_features = np.asarray(text_tokens)
        del emb_model
    else:
        x_vocab = mp.get('vocabs').get('token')
        for t in input_texts:
            tokens = [x_vocab[t] if t in x_vocab else len(
                x_vocab) + 1 for t in t.lower().split()]
            text_tokens.append(tokens)
        token_features, _ = get_paddedXY_sequence(
            text_tokens, [], sentence_length=sentence_len, shuffle=False)
    input_features.append(TaggedTextSequence(
        sentence_len, token_features, vec_input=mp['use_embeddings']))

    if mp['pos'] is True:
        pos_vocab = mp['vocabs']['pos']
        nlp = spacy.load('en')
        pos_tokens = []
        for doc in [nlp(t) for t in input_texts]:
            vec = [pos_vocab[t.tag_.lower()] if t.tag_.lower(
            ) in pos_vocab else len(pos_vocab) + 1 for t in doc]
            pos_tokens.append(vec)
        pos_features, _ = get_paddedXY_sequence(
            pos_tokens, [], sentence_length=sentence_len, shuffle=False)
        input_features.append(TaggedTextSequence(
            sentence_len, pos_features, vec_input=False))

    if len(input_features) > 1:
        x = MultiSequenceDataIterator(input_features, ignore_y=True)
    else:
        x = input_features[0]

    model = Model(model_file)
    model.initialize(dataset=x)
    preds = model.get_outputs(x).argmax(2)

    # print inferred tags and original text
    print_nps = args.print_only_nps
    rev_y = {v + 1: k for k, v in mp['y_vocab'].items()}
    for sen, text in zip(preds, input_texts):
        text_tokens = text.lower().split()
        text_tags = [rev_y[i] for i in sen if i > 0]
        if print_nps is True:
            print(extract_nps(text_tokens, text_tags))
        else:
            print(list(zip(text_tokens, text_tags)))
