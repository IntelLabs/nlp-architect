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

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from nlp_architect.models.ner_crf import NERCRF
from nlp_architect.utils.io import validate_existing_filepath
# from nlp_architect.utils.text import SpacyInstance

# nlp = SpacyInstance(disable=['tagger', 'ner', 'parser', 'vectors', 'textcat'])


def read_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=validate_existing_filepath, required=True,
                        help='Path of model weights')
    parser.add_argument('--model_info_path', type=validate_existing_filepath, required=True,
                        help='Path of model topology')
    input_args = parser.parse_args()
    return input_args


def load_saved_model():
    ner_model = NERCRF()
    ner_model.build(model_info['sentence_len'],
                    model_info['word_len'],
                    model_info['num_of_labels'],
                    model_info['word_vocab'],
                    model_info['vocab_size'],
                    model_info['char_vocab_size'],
                    word_embedding_dims=model_info['word_embedding_dims'],
                    char_embedding_dims=model_info['char_embedding_dims'],
                    word_lstm_dims=model_info['word_lstm_dims'],
                    tagger_lstm_dims=model_info['tagger_lstm_dims'],
                    dropout=model_info['dropout'],
                    external_embedding_model=model_info['external_embedding_model'])
    ner_model.load(args.model_path)
    return ner_model


def process_text(text):
    input_text = ' '.join(text.strip().split())
    return [text_arr for text_arr in input_text]


def encode_word(word):
    return model_info['word_vocab'].get(word, 1.0)


def encode_word_chars(word):
    return [model_info['char_vocab'].get(c, 1.0) for c in word]


def encode_input(text_arr):
    sentence = []
    sentence_chars = []
    for word in text_arr:
        sentence.append(encode_word(word))
        sentence_chars.append(encode_word_chars(word))
    encoded_sentence = pad_sequences([np.asarray(sentence)], maxlen=model_info['sentence_len'])
    chars_padded = pad_sequences(sentence_chars, maxlen=model_info['word_len'])
    if model_info['sentence_len'] - chars_padded.shape[0] > 0:
        chars_padded = np.concatenate((np.zeros((model_info['sentence_len'] -
                                                 chars_padded.shape[0], model_info['word_len'])),
                                       chars_padded))
    encoded_chars = chars_padded.reshape(1, model_info['sentence_len'], model_info['word_len'])
    return encoded_sentence, encoded_chars


def pretty_print(text, tags):
    tags_str = [model_info['labels_id_to_word'].get(t, None) for t in tags[0]][-len(text):]
    print(' '.join(['%-{}s'.format(max(len(t), len(tg)) + 1) % t
                    for t, tg in zip(text, tags_str)]))
    print(' '.join(['%-{}s'.format(max(len(t), len(tg)) + 1) % tg
                    for t, tg in zip(text, tags_str)]))


def run_interactive():
    model = load_saved_model()
    while True:
        text = input('Enter sentence >> ')
        text_arr = process_text(text)
        words, chars = encode_input(text_arr)
        tags = model.predict([words, chars])
        tags = tags.argmax(2)
        pretty_print(text_arr, tags)


if __name__ == '__main__':
    args = read_input_args()
    with open(args.model_info_path, 'rb') as fp:
        model_info = pickle.load(fp)
    assert model_info is not None, 'No model topology information loaded'
    run_interactive()
