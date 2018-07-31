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

from tensorflow import keras

from nlp_architect.contrib.keras.callbacks import ConllCallback
from nlp_architect.data.sequential_tagging import CONLL2000
from nlp_architect.models.chunker import SequenceChunker
from nlp_architect.utils.embedding import load_word_embeddings, get_embedding_matrix
from nlp_architect.utils.io import validate_existing_filepath, validate_parent_exists, validate, \
    validate_existing_directory
from nlp_architect.utils.metrics import get_conll_scores


def create_argument_parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--data_dir', type=validate_existing_directory,
                         help='Path to directory containing CONLL2000 files')
    _parser.add_argument('--embedding_model', type=validate_existing_filepath,
                         help='Word embedding model path (GloVe/Fasttext/textual)')
    _parser.add_argument('--sentence_length', default=50, type=int,
                         help='Maximum sentence length')
    _parser.add_argument('--feature_size', default=100, type=int,
                         help='Feature vector size (in embedding and LSTM layers)')
    _parser.add_argument('--use_gpu', default=False, action='store_true',
                         help='use GPU backend (CUDNN enabled)')
    _parser.add_argument('-b', default=10, type=int,
                         help='batch size')
    _parser.add_argument('-e', default=10, type=int,
                         help='number of epochs run fit model')
    _parser.add_argument('--model_name', default='chunker_model', type=str,
                         help='Model name (used for saving the model)')
    _parser.add_argument('--print_np', default=False, action='store_true',
                         help='Print only Noun Phrase (NP) tags accuracy')
    return _parser


def _save_model():
    model_params = {'word_vocab': dataset.word_vocab,
                    'pos_vocab': dataset.pos_vocab,
                    'chunk_vocab': dataset.chunk_vocab}
    with open(settings_path, 'wb') as fp:
        pickle.dump(model_params, fp)
    model.save(model_path)


if __name__ == '__main__':
    # read input args and validate
    parser = create_argument_parser()
    args = parser.parse_args()
    validate((args.sentence_length, int, 1, 1000))
    validate((args.feature_size, int, 1, 10000))
    validate((args.b, int, 1, 100000))
    validate((args.e, int, 1, 100000))
    model_path = path.join(path.dirname(path.realpath(__file__)),
                           '{}.h5'.format(str(args.model_name)))
    settings_path = path.join(path.dirname(path.realpath(__file__)),
                              '{}.params'.format(str(args.model_name)))
    validate_parent_exists(model_path)

    # load dataset and get tokens/chunks/pos tags
    dataset = CONLL2000(data_path=args.data_dir, sentence_length=args.sentence_length)
    words_train, pos_train, chunk_train = dataset.train_set
    words_test, pos_test, chunk_test = dataset.test_set

    # get label sizes, transform y's into 1-hot encoding
    chunk_labels = len(dataset.chunk_vocab) + 1
    pos_labels = len(dataset.pos_vocab) + 1
    word_vocab_size = len(dataset.word_vocab) + 2
    pos_train = keras.utils.to_categorical(pos_train, num_classes=pos_labels)
    chunk_train = keras.utils.to_categorical(chunk_train, num_classes=chunk_labels)
    pos_test = keras.utils.to_categorical(pos_test, num_classes=pos_labels)
    chunk_test = keras.utils.to_categorical(chunk_test, num_classes=chunk_labels)

    # build model with input parameters
    model = SequenceChunker(use_gpu=args.use_gpu)
    model.build(word_vocab_size,
                pos_labels,
                chunk_labels,
                feature_size=args.feature_size)

    # initialize word embedding if external model selected
    if args.embedding_model is not None:
        embedding_model, _ = load_word_embeddings(args.embedding_model)
        embedding_mat = get_embedding_matrix(embedding_model, dataset.word_vocab)
        model.load_embedding_weights(embedding_mat)

    # train the model
    chunk_f1_cb = ConllCallback(words_test, chunk_test, dataset.chunk_vocab.vocab, batch_size=64)
    model.fit(words_train, [pos_train, chunk_train],
              epochs=args.e, batch_size=args.b, callbacks=[chunk_f1_cb])
    # save model
    _save_model()

    # print evaluation metric
    model.chunk_inference_mode()
    chunk_pred = model.predict(words_test, 64)
    _, _, chunk_test = dataset.test_set

    res = get_conll_scores(chunk_pred, chunk_test, dataset.chunk_vocab.reverse_vocab())
    if args.print_np is True:
        print('NP F1: {}'.format(res[1]['NP'][-1]))
    else:
        print('Chunk F1: {}'.format(res[0][-1]))
