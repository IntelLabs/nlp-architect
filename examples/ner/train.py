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
import pprint
from os import path
from os.path import basename

from keras.utils import to_categorical

from nlp_architect.contrib.keras.callbacks import ConllCallback
from nlp_architect.data.sequential_tagging import SequentialTaggingDataset
from nlp_architect.models.ner_crf import NERCRF
from nlp_architect.utils.io import validate_existing_filepath, validate_parent_exists, validate
from nlp_architect.utils.metrics import get_conll_scores


def read_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--e', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--train_file', type=validate_existing_filepath, required=True,
                        help='Train file (sequential tagging dataset format)')
    parser.add_argument('--test_file', type=validate_existing_filepath, required=True,
                        help='Test file (sequential tagging dataset format)')
    parser.add_argument('--tag_num', type=int, default=4,
                        help='Entity labels tab number in train/test files')
    parser.add_argument('--sentence_length', type=int, default=30,
                        help='Max sentence length')
    parser.add_argument('--word_length', type=int, default=20,
                        help='Max word length in characters')
    parser.add_argument('--word_embedding_dims', type=int, default=100,
                        help='Word features embedding dimension size')
    parser.add_argument('--character_embedding_dims', type=int, default=25,
                        help='Character features embedding dimension size')
    parser.add_argument('--char_features_lstm_dims', type=int, default=25,
                        help='Character feature extractor LSTM dimension size')
    parser.add_argument('--entity_tagger_lstm_dims', type=int, default=100,
                        help='Entity tagger LSTM dimension size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--embedding_model', type=validate_existing_filepath,
                        help='Path to external word embedding model file')
    parser.add_argument('--model_path', type=str, default='model.h5',
                        help='Path for saving model weights')
    parser.add_argument('--model_info_path', type=str,
                        help='Path for saving model topology')
    input_args = parser.parse_args()
    validate_input_args(input_args)
    return input_args


def validate_input_args(args):
    validate((args.b, int, 1, 100000))
    validate((args.e, int, 1, 100000))
    validate((args.tag_num, int, 1, 1000))
    validate((args.sentence_length, int, 1, 10000))
    validate((args.word_length, int, 1, 100))
    validate((args.word_embedding_dims, int, 1, 10000))
    validate((args.character_embedding_dims, int, 1, 1000))
    validate((args.char_features_lstm_dims, int, 1, 10000))
    validate((args.entity_tagger_lstm_dims, int, 1, 10000))
    validate((args.dropout, float, 0, 1))
    model_path = path.join(path.dirname(path.realpath(__file__)), str(args.model_path))
    validate_parent_exists(model_path)
    model_info_path = path.join(path.dirname(path.realpath(__file__)), str(args.model_info_path))
    validate_parent_exists(model_info_path)


def run_ner(
        train_file,
        test_file,
        embedding_model,
        batch_size=10,
        epoch=50,
        tag_num=2,
        sentence_length=30,
        word_length=20,
        word_embedding_dims=100,
        character_embedding_dims=25,
        char_features_lstm_dims=25,
        entity_tagger_lstm_dims=100,
        dropout=0.2,
):
    # load dataset and parameters
    dataset = SequentialTaggingDataset(
        train_file, test_file, max_sentence_length=sentence_length, max_word_length=word_length, tag_field_no=tag_num)

    # get the train and test data sets
    x_train, x_char_train, y_train = dataset.train
    x_test, x_char_test, y_test = dataset.test

    num_y_labels = len(dataset.y_labels) + 1
    vocabulary_size = dataset.word_vocab_size + 1
    char_vocabulary_size = dataset.char_vocab_size + 1

    y_test = to_categorical(y_test, num_y_labels)
    y_train = to_categorical(y_train, num_y_labels)

    ner_model = NERCRF()
    ner_model.build(
        sentence_length,
        word_length,
        num_y_labels,
        dataset.word_vocab,
        vocabulary_size,
        char_vocabulary_size,
        word_embedding_dims=word_embedding_dims,
        char_embedding_dims=character_embedding_dims,
        word_lstm_dims=char_features_lstm_dims,
        tagger_lstm_dims=entity_tagger_lstm_dims,
        dropout=dropout,
        external_embedding_model=embedding_model)

    conll_cb = ConllCallback([x_test, x_char_test], y_test, dataset.y_labels,
                             batch_size=batch_size)

    ner_model.fit(
        x=[x_train, x_char_train],
        y=y_train,
        batch_size=batch_size,
        epochs=epoch,
        callbacks=[conll_cb],
        validation=([x_test, x_char_test], y_test))

    # running predictions
    predictions = ner_model.predict(x=[x_test, x_char_test], batch_size=1)
    eval = get_conll_scores(predictions, y_test, {v: k for k, v in dataset.y_labels.items()})
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(eval)

    # saving model
    # ner_model.save(model_path)
    with open('model-info' + '-' + basename(train_file), 'wb') as fp:
        info = {
            'sentence_len': sentence_length,
            'word_len': word_length,
            'num_of_labels': num_y_labels,
            'labels_id_to_word': {v: k for k, v in dataset.y_labels.items()},
            'word_vocab': dataset.word_vocab,
            'vocab_size': vocabulary_size,
            'char_vocab_size': char_vocabulary_size,
            'char_vocab': dataset.char_vocab,
            'word_embedding_dims': word_embedding_dims,
            'char_embedding_dims': character_embedding_dims,
            'word_lstm_dims': char_features_lstm_dims,
            'tagger_lstm_dims': entity_tagger_lstm_dims,
            'dropout': dropout,
            'external_embedding_model': embedding_model,
            'train_file': train_file,
            'test_file': test_file,
            'eval': eval
        }
        pickle.dump(info, fp)


if __name__ == '__main__':
    # parse the input
    args = read_input_args()

    # load dataset and parameters
    dataset = SequentialTaggingDataset(args.train_file, args.test_file,
                                       max_sentence_length=args.sentence_length,
                                       max_word_length=args.word_length,
                                       tag_field_no=args.tag_num)

    # get the train and test data sets
    x_train, x_char_train, y_train = dataset.train
    x_test, x_char_test, y_test = dataset.test

    num_y_labels = len(dataset.y_labels) + 1
    vocabulary_size = dataset.word_vocab_size + 1
    char_vocabulary_size = dataset.char_vocab_size + 1

    y_test = to_categorical(y_test, num_y_labels)
    y_train = to_categorical(y_train, num_y_labels)

    ner_model = NERCRF()
    ner_model.build(args.sentence_length,
                    args.word_length,
                    num_y_labels,
                    dataset.word_vocab,
                    vocabulary_size,
                    char_vocabulary_size,
                    word_embedding_dims=args.word_embedding_dims,
                    char_embedding_dims=args.character_embedding_dims,
                    word_lstm_dims=args.char_features_lstm_dims,
                    tagger_lstm_dims=args.entity_tagger_lstm_dims,
                    dropout=args.dropout,
                    external_embedding_model=args.embedding_model)

    conll_cb = ConllCallback([x_test, x_char_test], y_test, dataset.y_labels, batch_size=args.b)

    ner_model.fit(x=[x_train, x_char_train], y=y_train,
                  batch_size=args.b,
                  epochs=args.e,
                  callbacks=[conll_cb],
                  validation=([x_test, x_char_test], y_test))

    # saving model
    ner_model.save(args.model_path)
    with open(args.model_info_path, 'wb') as fp:
        info = {
            'sentence_len': args.sentence_length,
            'word_len': args.word_length,
            'num_of_labels': num_y_labels,
            'labels_id_to_word': {v: k for k, v in dataset.y_labels.items()},
            'word_vocab': dataset.word_vocab,
            'vocab_size': vocabulary_size,
            'char_vocab_size': char_vocabulary_size,
            'char_vocab': dataset.char_vocab,
            'word_embedding_dims': args.word_embedding_dims,
            'char_embedding_dims': args.character_embedding_dims,
            'word_lstm_dims': args.char_features_lstm_dims,
            'tagger_lstm_dims': args.entity_tagger_lstm_dims,
            'dropout': args.dropout,
            'external_embedding_model': args.embedding_model
        }
        pickle.dump(info, fp)

    # running predictions
    predictions = ner_model.predict(x=[x_test, x_char_test], batch_size=1)
    eval = get_conll_scores(predictions, y_test, {v: k for k, v in dataset.y_labels.items()})
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(eval)
