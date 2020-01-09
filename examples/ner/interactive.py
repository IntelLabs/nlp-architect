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

from nlp_architect.models.ner_crf import NERCRF
from nlp_architect.utils.generic import pad_sentences
from nlp_architect.utils.io import validate_existing_filepath
from nlp_architect.utils.text import SpacyInstance

nlp = SpacyInstance(disable=["tagger", "ner", "parser", "vectors", "textcat"])


def read_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=validate_existing_filepath, required=True, help="Path of model weights"
    )
    parser.add_argument(
        "--model_info_path",
        type=validate_existing_filepath,
        required=True,
        help="Path of model topology",
    )
    input_args = parser.parse_args()
    return input_args


def load_saved_model():
    ner_model = NERCRF()
    ner_model.load(args.model_path)
    return ner_model


def process_text(doc):
    input_text = " ".join(doc.strip().split())
    return nlp.tokenize(input_text)


def vectorize(doc, w_vocab, c_vocab):
    words = np.asarray([w_vocab[w.lower()] if w.lower() in w_vocab else 1 for w in doc]).reshape(
        1, -1
    )
    sentence_chars = []
    for w in doc:
        word_chars = []
        for c in w:
            if c in c_vocab:
                _cid = c_vocab[c]
            else:
                _cid = 1
            word_chars.append(_cid)
        sentence_chars.append(word_chars)
    sentence_chars = np.expand_dims(pad_sentences(sentence_chars, model.word_length), axis=0)
    return words, sentence_chars


if __name__ == "__main__":
    args = read_input_args()
    with open(args.model_info_path, "rb") as fp:
        model_info = pickle.load(fp)
    assert model_info is not None, "No model topology information loaded"
    model = load_saved_model()
    word_vocab = model_info["word_vocab"]
    y_vocab = {v: k for k, v in model_info["y_vocab"].items()}
    char_vocab = model_info["char_vocab"]
    while True:
        text = input("Enter sentence >> ")
        text_arr = process_text(text)
        doc_vec = vectorize(text_arr, word_vocab, char_vocab)
        seq_len = np.array([len(text_arr)]).reshape(-1, 1)
        inputs = list(doc_vec)
        # pylint: disable=no-member
        if model.crf_mode == "pad":
            inputs = list(doc_vec) + [seq_len]
        doc_ner = model.predict(inputs, batch_size=1).argmax(2).flatten()
        ners = [y_vocab.get(n, None) for n in doc_ner]
        for t, n in zip(text_arr, ners):
            print("{}\t{}\t".format(t, n))
        print()
