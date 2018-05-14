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
import os
from os import path

from keras.callbacks import ModelCheckpoint
from nlp_architect.contrib.keras.callbacks import ConllCallback
from nlp_architect.data.intent_datasets import SNIPS
from nlp_architect.models.intent_extraction import EncDecIntentModel
from nlp_architect.utils.io import validate_existing_directory, validate, \
    validate_existing_filepath, validate_parent_exists
from nlp_architect.utils.metrics import get_conll_scores


def validate_input_args():
    global model_path
    validate((args.b, int, 1, 100000000))
    validate((args.e, int, 1, 100000000))
    validate((args.sentence_length, int, 1, 10000))
    validate((args.token_emb_size, int, 1, 10000))
    validate((args.lstm_hidden_size, int, 1, 10000))
    validate((args.encoder_depth, int, 1, 10))
    validate((args.decoder_depth, int, 1, 10))
    validate((args.save_epochs, int, 1, 1000))
    validate((args.encoder_dropout, float, 0, 1))
    validate((args.decoder_dropout, float, 0, 1))
    model_path = path.join(path.dirname(path.realpath(__file__)), str(args.model_path))
    validate_parent_exists(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=10,
                        help='Batch size')
    parser.add_argument('-e', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--dataset_path', type=validate_existing_directory, required=True,
                        help='dataset directory')
    parser.add_argument('--sentence_length', type=int, default=30,
                        help='Max sentence length')
    parser.add_argument('--token_emb_size', type=int, default=100,
                        help='Token features embedding vector size')
    parser.add_argument('--lstm_hidden_size', type=int, default=150,
                        help='Encoder LSTM hidden size')
    parser.add_argument('--encoder_depth', type=int, default=1,
                        help='Encoder LSTM depth')
    parser.add_argument('--decoder_depth', type=int, default=1,
                        help='Decoder LSTM depth')
    parser.add_argument('--encoder_dropout', type=float, default=0.5,
                        help='Encoder dropout value')
    parser.add_argument('--decoder_dropout', type=float, default=0.5,
                        help='Decoder dropout value')
    parser.add_argument('--embedding_model', type=validate_existing_filepath,
                        help='Path to word embedding model file')
    parser.add_argument('--full_eval', action='store_true', default=False,
                        help='Print full slot tagging statistics instead of global P/R/F1')
    parser.add_argument('--restore', action='store_true', default=False,
                        help='Restore model weights from model path')
    parser.add_argument('--model_path', type=str, default='model.h5',
                        help='Model file path')
    parser.add_argument('--save_epochs', type=int, default=1,
                        help='Number of epochs to run between model saves')
    args = parser.parse_args()
    validate_input_args()
    dataset = SNIPS(path=args.dataset_path,
                    sentence_length=args.sentence_length,
                    embedding_model=args.embedding_model,
                    embedding_size=args.token_emb_size)

    train_x, _, train_i, train_y = dataset.train_set
    test_x, _, test_i, test_y = dataset.test_set

    model = EncDecIntentModel()

    if args.restore and os.path.exists(model_path):
        print('Loading model weights and continuing with training ..')
        model.load(model_path)
    else:
        print('Creating new model, starting to train from scratch')
        model.build(args.sentence_length,
                    dataset.vocab_size,
                    dataset.label_vocab_size,
                    args.token_emb_size,
                    args.encoder_depth,
                    args.decoder_depth,
                    args.lstm_hidden_size,
                    args.encoder_dropout,
                    args.decoder_depth,
                    args.embedding_model)

    conll_cb = ConllCallback(test_x, test_y, dataset.labels_vocab, batch_size=args.b)
    cp_cb = ModelCheckpoint(model_path, verbose=1, period=args.save_epochs)

    # train model
    model.fit(x=train_x, y=train_y,
              batch_size=args.b, epochs=args.e,
              validation=(test_x, test_y),
              callbacks=[conll_cb, cp_cb])
    print('Training done.')

    # test performance
    predictions = model.predict(test_x, batch_size=args.b)
    eval = get_conll_scores(predictions, test_y, {
        v: k for k, v in dataset.labels_vocab.items()})
    if args.full_eval is True:
        print(eval)
    else:
        print(eval[0])
