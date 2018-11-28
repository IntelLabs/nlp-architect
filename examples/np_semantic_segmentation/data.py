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

# pylint: disable=global-statement

import argparse
import csv
import math
import os
from multiprocessing import Pool

import feature_extraction as fe
import numpy
from tqdm import tqdm

from nlp_architect.utils.io import validate_existing_filepath, validate_parent_exists, \
    validate_proxy_path

wordnet = None
wikidata = None
word2vec = None


def build_feature_vector(np):
    """
    Build a feature vector for the given noun-phrase. the size of the
    vector = (4 + (#WORDS X 300)) = 1506. ==> (the number of words in a noun-phrase X size
    of the word2vec(300)) + 4 additional features from external sources

    Args:
        np (str): a noun-phrase

    Returns:
        :obj:`np.ndarray`: the feature vector of the np
    """
    feature_vector = []
    # 1. find if np exist as an entity in WordNet
    wordnet_feature = find_wordnet_entity(np)
    feature_vector.append(wordnet_feature)
    # 2. find if np exist as an entity in Wikidata
    wikidata_feature = find_wikidata_entity(np)
    feature_vector.append(wikidata_feature)
    # 3. word2vec score: from
    word2vec_distance = word2vec.get_similarity_score(np)
    feature_vector.append(word2vec_distance)
    for w in np.split(" "):
        feature_vector.extend(word2vec.get_word_embedding(w))
    return numpy.array(feature_vector)


def find_wordnet_entity(np):
    """
    extract WordNet indicator-feature (1 if exist in WordNet, else 0)

    Args:
        np (str): a noun-phrase

    Returns:
        int: 1 if exist in WordNet, else 0
    """
    candidates = expand_np_candidates(np, True)
    return wordnet.find_wordnet_existence(candidates)


def find_wikidata_entity(np):
    """
    extract Wikidata indicator-feature (1 if exist in Wikidata, else 0)

    Args:
        np (str): a noun-phrase

    Returns:
        int: 1 if exist in Wikidata, else 0
    """
    candidates = expand_np_candidates(np, True)
    return wikidata.find_wikidata_existence(candidates)


def expand_np_candidates(np, stemming):
    """
    Create all case-combination of the noun-phrase (nyc to NYC, israel to Israel etc.)

    Args:
        np (str): a noun-phrase
        stemming (bool): True if to add case-combinations of noun-phrases's stem

    Returns:
        list(str): All case-combination of the noun-phrase
    """
    candidates = []
    # create all case-combinations of np-> nyc to NYC, israel to Israel etc.
    candidates.extend(get_all_case_combinations(np))
    if stemming:
        # create all case-combinations of np's stem-> t-shirts to t-shirt etc.
        candidates.extend(get_all_case_combinations(fe.stem(np)))
    return candidates


def get_all_case_combinations(np):
    """
    Returns all case combinations for the noun-phrase (regular, upper, lower, title)
    Args:
        np (str): a noun-phrase
    Returns:
        list(str): List of all case combinations
    """
    candidates = [np, np.upper(), np.lower(), np.title()]
    return candidates


def write_to_csv(output, np_feature_vectors, np_dic, np_list):
    """
    Write data to csv file

    Args:
        output (str): output file path
        np_feature_vectors (:obj:`np.ndarray`): numpy vectors
        np_dic (dict): dict, keys: the noun phrase, value: the features
        np_list (list): features list
    """
    with open(output, 'w', encoding='utf-8') as out_file:
        writer = csv.writer(out_file, delimiter=',', quotechar='"')
        print("prepared data CSV file is saved in {0}".format(output))
        for i, _ in enumerate(np_feature_vectors):
            np_vector = np_feature_vectors[i]
            np_vector = numpy.append(np_vector, np_dic[np_list[i]])
            writer.writerow(np_vector)


def prepare_data(data_file, output_file, word2vec_file, http_prox=None, https_prox=None):
    """
    Extract for each noun-phrase a feature vector (W2V, WordNet, Wikidata, NPMI, UCI).
    Write the feature vectors to --output specifies local path

    Args:
        data_file(str): file_path to input data
        output_file(str): file_path to output processed data
        word2vec_file(str): file_path to word2vec model
        http_prox(str): http_proxy
        https_prox(str): https_proxy
    """
    # init_resources:
    global wordnet, wikidata, word2vec
    wordnet = fe.Wordnet()
    wikidata = fe.Wikidata(http_prox, https_prox)
    print("Start loading Word2Vec model (this might take a while...)")
    word2vec = fe.Word2Vec(word2vec_file)
    print("Finish loading feature extraction services")
    reader_list = read_csv_file_data(data_file)
    np_dic = {}
    np_list = []
    for row in reader_list:
        np_dic[row[0]] = row[1]
        np_list.append(row[0])
    p = Pool(10)
    np_feature_vectors = list(tqdm(p.imap(build_feature_vector, np_list),
                                   total=len(np_list)))  # , desc="np feature extraction status"))
    write_to_csv(output_file, np_feature_vectors, np_dic, np_list)


def read_csv_file_data(input_path):
    """
    Read csv file to a list

    Args:
        input_path (str): read csv file from this local file path

    Returns:
        list(str): A list where each item is a row in the csv file
    """
    # 1. read csv file
    if not os.path.isabs(input_path):
        # handle case using default value\relative paths
        input_path = os.path.join(os.path.dirname(__file__), input_path)
    with open(input_path, 'r', encoding='utf-8-sig') as input_file:
        reader = csv.reader((line.replace('\0', '') for line in input_file))
        reader_list = list(reader)
    return reader_list


def extract_y_labels(input_path):
    """
    Extract only the Labels of the data

    Args:
        input_path (str): read csv file from this local file path

    Returns:
        :obj:`np.ndarray`: A numpy array of the labels, each item is the label of the row
    """
    reader_list = read_csv_file_data(input_path)
    Y_labels_vec = []
    cntr = 0
    for line in reader_list:
        Y_label = line[-1]
        Y_labels_vec.insert(cntr, Y_label)
        cntr += 1
    y_train = Y_labels_vec
    y_train = numpy.array(y_train, dtype=numpy.int32)
    return y_train


class NpSemanticSegData:
    """
    Dataset for NP Semantic Segmentation Model

        Args:
            file_path (str): read data from this local file path

            train_to_test_ratio (:obj:`float`): the train-to-test ration of the dataset

            feature_vec_dim (:obj:`int`): the size of the feature vector for each noun-phrase
    """

    def __init__(self, file_path, train_to_test_ratio=0.8, feature_vec_dim=603):
        self.file_path = file_path
        self.feature_vec_dim = feature_vec_dim
        self.train_to_test_ratio = train_to_test_ratio
        self.is_y_labels = None
        self.y_labels = None
        self.data_set = self.load_data_to_array_iterator()

    def load_data_from_file(self):
        """
        Loads data from file_path to X_train, y_train, X_test, y_test numpy arrays

        Returns:
            tuple(:obj:`np.ndarray`): X_train, y_train, X_test, y_test numpy arrays
        """
        reader_list = read_csv_file_data(self.file_path)
        # count num of feature vectors
        num_feats = len(reader_list)
        # is_y_labels is for inference - if the inference data is labeled y_labels are extracted
        self.is_y_labels = len(reader_list[0]) == self.feature_vec_dim + 1
        X_feature_matrix = numpy.zeros((num_feats, self.feature_vec_dim))
        Y_labels_vec = []
        cntr = 0
        for line in reader_list:
            X_features = numpy.array(line[:self.feature_vec_dim])
            X_feature_matrix[cntr, :] = X_features

            if self.is_y_labels:
                Y_label = line[self.feature_vec_dim]
                Y_labels_vec.insert(cntr, Y_label)
            cntr += 1

        len_train = int(math.floor(num_feats * self.train_to_test_ratio))

        X_train = X_feature_matrix[0:len_train - 1]
        y_train = None
        if self.is_y_labels:
            y_train = Y_labels_vec[0:len_train - 1]
            y_train = numpy.array(y_train, dtype=numpy.int32)

        X_test = X_feature_matrix[len_train:]
        y_test = None
        if self.is_y_labels:
            y_test = Y_labels_vec[len_train:]
            y_test = numpy.array(y_test, dtype=numpy.int32)
        return X_train, y_train, X_test, y_test

    def load_data_to_array_iterator(self):
        """
        Load data into dict of 'train' and 'test' datasets

        Returns:
            dict: dict with train set & test_set (each is dict with X and y)
        """
        X_train, y_train, X_test, y_test = self.load_data_from_file()
        data_set = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        return data_set

    @property
    def train_set(self):
        """dict(:obj:`numpy.ndarray`): train set (X & y)"""
        return self.data_set['train']

    @property
    def train_set_x(self):
        """dict(:obj:`numpy.ndarray`): train set (X)"""
        return self.data_set['train']['X']

    @property
    def train_set_y(self):
        """dict(:obj:`numpy.ndarray`): train set (y)"""
        return self.data_set['train']['y']

    @property
    def test_set(self):
        """dict(:obj:`numpy.ndarray`): test set (X & y)"""
        return self.data_set['test']

    @property
    def test_set_x(self):
        """dict(:obj:`numpy.ndarray`): test set (X)"""
        return self.data_set['test']['X']

    @property
    def test_set_y(self):
        """dict(:obj:`numpy.ndarray`): test set (y)"""
        return self.data_set['test']['y']


def absolute_path(input_path):
    """
    Return input_path's absolute path

    Args:
        input_path(str): input_path

    Returns:
        str: absolute path
    """
    if isinstance(input_path, str):
        if not os.path.isabs(input_path):
            # handle case using default value\relative paths
            input_path = os.path.join(os.path.dirname(__file__), input_path)
    return input_path


if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser(
        description='Prepare data')
    parser.add_argument('--data', type=validate_existing_filepath,
                        help='path the CSV file where the raw dataset is saved')
    parser.add_argument('--output', type=validate_parent_exists,
                        help='path the CSV file where the prepared dataset will be saved')
    parser.add_argument('--w2v_path', type=validate_existing_filepath,
                        help='path to the word embedding\'s model')
    parser.add_argument('--http_proxy', help='system\'s http proxy',
                        type=validate_proxy_path, default=None)
    parser.add_argument('--https_proxy', help='system\'s https proxy',
                        type=validate_proxy_path, default=None)
    args = parser.parse_args()
    data_path = absolute_path(args.data)
    word2vec_path = args.w2v_path
    output_path = absolute_path(args.output)
    http_proxy = args.http_proxy
    https_proxy = args.https_proxy
    prepare_data(data_path, output_path, word2vec_path, http_proxy, https_proxy)
