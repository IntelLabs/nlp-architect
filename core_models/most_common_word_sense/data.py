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
# ****************************************************************************
"""
Prepare the datasets for Most Common Word Sense training
"""

import numpy as np
import pickle
import csv
import logging
import codecs
import gensim
import math
from neon.util.argparser import NeonArgparser
from sklearn.model_selection import train_test_split
from models.most_common_word_sense.feature_extraction import extract_features_envelope

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# -------------------------------------------------------------------------------------#
def read_gs_file(gs_file_name):
    # read gold std file

    file = codecs.open(gs_file_name, 'rU', 'utf-8')
    reader = csv.reader((line.replace('\0', '') for line in file))

    cntr = 0
    # 1. read csv file
    target_word_vec = []
    definition_vec = []
    hypernym_vec = []

    label_vec = []

    header_line_flag = True
    for line in reader:
        if line is not None:
            if header_line_flag:  # skip header line
                header_line_flag = False
                continue

            target_word_vec.insert(cntr, line[0].strip())
            definition_vec.insert(cntr, line[1])
            hypernym_vec.insert(cntr, line[2])
            label_vec.insert(cntr, line[3])
            cntr = cntr + 1

    file.close()

    return target_word_vec, definition_vec, hypernym_vec, label_vec


# -------------------------------------------------------------------------------------#


def read_inference_input_examples_file(input_examples_file):
    # read gold std file

    file = codecs.open(input_examples_file, 'rU', 'utf-8')
    reader = csv.reader((line.replace('\0', '') for line in file))

    cntr = 0
    # 1. read csv file
    target_word_vec = []
    header_line_flag = True
    for line in reader:
        if line is not None:
            if header_line_flag:  # skip header line
                header_line_flag = False
                continue
            target_word_vec.insert(cntr, line[0])

            cntr = cntr + 1

    file.close()

    return target_word_vec


if __name__ == "__main__":

    parser = NeonArgparser(__doc__)

    parser.add_argument('--gold_standard_file',
                        default='data/goldStd.csv',
                        help='path to gold standard file')
    parser.add_argument('--word_embedding_model_file',
                        default='pretrained_models/GoogleNews-vectors-negative300.bin',
                        help='path to the word embedding\'s model')

    parser.add_argument('--training_to_validation_size_ratio', default='0.8',
                        help='ratio between training and validation size')
    parser.add_argument('--data_set_file', default='data/data_set.pkl',
                        help='path the file where the train, valid and test sets are stored')

    args = parser.parse_args()

    # training set
    X_train = []
    y_train = []

    # validation set
    X_valid = []
    y_valid = []

    # 1. read GS file
    [target_word_vec, definition_vec, hypernym_vec, label_vec] = read_gs_file(args.gold_standard_file)
    logger.info("finished reading GS file")

    # 2. Load pre-trained word embeddings model.
    word_embeddings_model = gensim.models.KeyedVectors.load_word2vec_format(args.word_embedding_model_file, binary=True)
    logger.info("finished loading word embeddings model")

    feat_vec_dim = 603
    num_samples = len(list(target_word_vec))

    x_feature_matrix = np.zeros((num_samples, feat_vec_dim))
    y_labels_vec = []

    i = 0
    cntr = 0
    for target_word in target_word_vec:
        InputWord = target_word_vec[i]
        definition = definition_vec[i]
        hypslist = hypernym_vec[i]
        label = label_vec[i]

        [valid_w2v_flag, definition_sim_cbow, definition_sim, hyps_sim, target_word_emb, definition_sentence_emb_cbow] \
            = extract_features_envelope(InputWord, definition, hypslist, word_embeddings_model)

        if definition_sim_cbow != 0 and definition_sim != 0 and valid_w2v_flag is True:
            feature_vec = np.array([definition_sim_cbow, definition_sim, hyps_sim])
            feature_vec = np.concatenate((feature_vec, target_word_emb), 0)
            feature_vec = np.concatenate((feature_vec, definition_sentence_emb_cbow), 0)
            X_features = np.array(feature_vec)
            x_feature_matrix[cntr, :] = X_features

            #           binary classifier = 2 categories
            y_vec = np.zeros(2, 'uint8')
            y_vec[int(label)] = 1
            y_labels_vec.append(y_vec)

            cntr = cntr + 1

        i = i + 1

    logger.info("finished feature extraction")
    x_feature_matrix = x_feature_matrix[0:len(y_labels_vec), 0:feat_vec_dim]

    # split between train and valid sets
    X_train1, X_valid1, y_train1, y_valid1 = train_test_split(
        x_feature_matrix, y_labels_vec, train_size=math.ceil(
                                                             num_samples*float(args.training_to_validation_size_ratio)))

    X_train.extend(X_train1)
    X_valid.extend(X_valid1)
    y_train.extend(y_train1)
    y_valid.extend(y_valid1)

    logger.info('training set size: ' + repr(len(y_train)))
    logger.info('validation set size: ' + repr(len(y_valid)))

    # store data on file
    data_out = dict()

    data_out['X_train'] = X_train
    data_out['X_valid'] = X_valid
    data_out['y_train'] = y_train
    data_out['y_valid'] = y_valid

    with open(args.data_set_file, 'wb') as fp:
        pickle.dump(data_out, fp)
