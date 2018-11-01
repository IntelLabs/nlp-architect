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

import numpy as np

from nlp_architect.models.intent_extraction import MultiTaskIntentModel, Seq2SeqIntentModel
from nlp_architect.utils.generic import pad_sentences
from nlp_architect.utils.io import validate_existing_filepath
from nlp_architect.utils.text import SpacyInstance

nlp = SpacyInstance(disable=['tagger', 'ner', 'parser', 'vectors', 'textcat'])


def read_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=validate_existing_filepath, required=True,
                        help='Path of model weights')
    parser.add_argument('--model_info_path', type=validate_existing_filepath, required=True,
                        help='Path of model topology')
    input_args = parser.parse_args()
    return input_args


def load_saved_model():
    if model_type == 'seq2seq':
        model = Seq2SeqIntentModel()
    else:
        model = MultiTaskIntentModel()
    model.load(args.model_path)
    return model


def process_text(text):
    input_text = ' '.join(text.strip().split())
    return nlp.tokenize(input_text)


def vectorize(doc, vocab, char_vocab=None):
    words = np.asarray([vocab[w.lower()] if w.lower() in vocab else 1 for w in doc])\
        .reshape(1, -1)
    if char_vocab is not None:
        sentence_chars = []
        for w in doc:
            word_chars = []
            for c in w:
                if c in char_vocab:
                    _cid = char_vocab[c]
                else:
                    _cid = 1
                word_chars.append(_cid)
            sentence_chars.append(word_chars)
        sentence_chars = np.expand_dims(pad_sentences(sentence_chars, model.word_length), axis=0)
        return [words, sentence_chars]
    else:
        return words


if __name__ == '__main__':
    args = read_input_args()
    with open(args.model_info_path, 'rb') as fp:
        model_info = pickle.load(fp)
    assert model_info is not None, 'No model topology information loaded'
    model_type = model_info['type']
    model = load_saved_model()
    word_vocab = model_info['word_vocab']
    tags_vocab = {v: k for k, v in model_info['tags_vocab'].items()}
    if model_type == 'mtl':
        char_vocab = model_info['char_vocab']
        intent_vocab = {v: k for k, v in model_info['intent_vocab'].items()}
    while True:
        text = input('Enter sentence >> ')
        text_arr = process_text(text)
        if model_type == 'mtl':
            doc_vec = vectorize(text_arr, word_vocab, char_vocab)
            intent, tags = model.predict(doc_vec, batch_size=1)
            intent = int(intent.argmax(1).flatten())
            print('Detected intent type: {}'.format(intent_vocab.get(intent, None)))
        else:
            doc_vec = vectorize(text_arr, word_vocab, None)
            tags = model.predict(doc_vec, batch_size=1)
        tags = tags.argmax(2).flatten()
        tag_str = [tags_vocab.get(n, None) for n in tags]
        for t, n in zip(text_arr, tag_str):
            print('{}\t{}\t'.format(t, n))
        print()
