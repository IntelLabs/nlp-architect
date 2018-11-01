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
import pickle
from os import path

import numpy as np
from tensorflow import keras

from nlp_architect.contrib.tensorflow.python.keras.callbacks import ConllCallback
from nlp_architect.data.sequential_tagging import SequentialTaggingDataset
from nlp_architect.models.ner_crf import NERCRF
from nlp_architect.utils.embedding import get_embedding_matrix, load_word_embeddings
from nlp_architect.utils.io import validate, validate_existing_filepath, validate_parent_exists
from nlp_architect.utils.metrics import get_conll_scores


def read_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=10,
                        help='Batch size')
    parser.add_argument('-e', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--train_file', type=validate_existing_filepath, required=True,
                        help='Train file (sequential tagging dataset format)')
    parser.add_argument('--test_file', type=validate_existing_filepath, required=True,
                        help='Test file (sequential tagging dataset format)')
    parser.add_argument('--tag_num', type=int, default=2,
                        help='Entity labels tab number in train/test files')
    parser.add_argument('--sentence_length', type=int, default=50,
                        help='Max sentence length')
    parser.add_argument('--word_length', type=int, default=12,
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
    parser.add_argument('--model_info_path', type=str, default='model_info.dat',
                        help='Path for saving model topology')
    parser.add_argument('--use_cudnn', default=False, action='store_true',
                        help='use CUDNN based LSTM cells')
    input_args = parser.parse_args()
    validate_input_args(input_args)
    return input_args


def validate_input_args(input_args):
    validate((input_args.b, int, 1, 100000))
    validate((input_args.e, int, 1, 100000))
    validate((input_args.tag_num, int, 1, 1000))
    validate((input_args.sentence_length, int, 1, 10000))
    validate((input_args.word_length, int, 1, 100))
    validate((input_args.word_embedding_dims, int, 1, 10000))
    validate((input_args.character_embedding_dims, int, 1, 1000))
    validate((input_args.char_features_lstm_dims, int, 1, 10000))
    validate((input_args.entity_tagger_lstm_dims, int, 1, 10000))
    validate((input_args.dropout, float, 0, 1))
    model_path = path.join(path.dirname(path.realpath(__file__)), str(input_args.model_path))
    validate_parent_exists(model_path)
    model_info_path = path.join(path.dirname(path.realpath(__file__)),
                                str(input_args.model_info_path))
    validate_parent_exists(model_info_path)


if __name__ == '__main__':
    # parse the input
    args = read_input_args()

    # load dataset and parameters
    dataset = SequentialTaggingDataset(args.train_file, args.test_file,
                                       max_sentence_length=args.sentence_length,
                                       max_word_length=args.word_length,
                                       tag_field_no=args.tag_num)

    # get the train and test data sets
    x_train, x_char_train, y_train = dataset.train_set
    x_test, x_char_test, y_test = dataset.test_set

    num_y_labels = len(dataset.y_labels) + 1
    vocabulary_size = dataset.word_vocab_size
    char_vocabulary_size = dataset.char_vocab_size

    y_test = keras.utils.to_categorical(y_test, num_y_labels)
    y_train = keras.utils.to_categorical(y_train, num_y_labels)

    ner_model = NERCRF(use_cudnn=args.use_cudnn)
    ner_model.build(args.word_length,
                    num_y_labels,
                    vocabulary_size,
                    char_vocabulary_size,
                    word_embedding_dims=args.word_embedding_dims,
                    char_embedding_dims=args.character_embedding_dims,
                    word_lstm_dims=args.char_features_lstm_dims,
                    tagger_lstm_dims=args.entity_tagger_lstm_dims,
                    dropout=args.dropout)

    # initialize word embedding if external model selected
    if args.embedding_model is not None:
        embedding_model, _ = load_word_embeddings(args.embedding_model)
        embedding_mat = get_embedding_matrix(embedding_model, dataset.word_vocab)
        ner_model.load_embedding_weights(embedding_mat)

    train_inputs = [x_train, x_char_train]
    test_inputs = [x_test, x_char_test]
    if not args.use_cudnn:
        train_inputs.append(np.sum(np.not_equal(x_train, 0), axis=-1).reshape((-1, 1)))
        test_inputs.append(np.sum(np.not_equal(x_test, 0), axis=-1).reshape((-1, 1)))

    conll_cb = ConllCallback(test_inputs, y_test, dataset.y_labels.vocab,
                             batch_size=args.b)
    ner_model.fit(x=train_inputs, y=y_train,
                  batch_size=args.b,
                  epochs=args.e,
                  callbacks=[conll_cb],
                  validation=(test_inputs, y_test))

    # saving model
    ner_model.save(args.model_path)
    with open(args.model_info_path, 'wb') as fp:
        info = {
            'y_vocab': dataset.y_labels.vocab,
            'word_vocab': dataset.word_vocab.vocab,
            'char_vocab': dataset.char_vocab.vocab
        }
        pickle.dump(info, fp)

    # running predictions
    predictions = ner_model.predict(x=test_inputs, batch_size=args.b)
    eval = get_conll_scores(predictions, y_test, {v: k for k, v in dataset.y_labels.vocab.items()})
    print(eval)
