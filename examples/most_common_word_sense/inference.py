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
Most Common Word Sense Inference module.
"""
import argparse
import logging
import gensim
import numpy as np
from nltk.corpus import wordnet as wn
from termcolor import colored

from examples.most_common_word_sense.feature_extraction import extract_synset_data, \
    extract_features_envelope
from examples.most_common_word_sense.prepare_data import read_inference_input_examples_file
from nlp_architect.models.most_common_word_sense import MostCommonWordSense
from nlp_architect.utils.io import validate_existing_filepath, check_size

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def wsd_classify(x_test, y_test=None):
    """
    classifiy target word. output all word senses ranked according to the most probable sense

    Args:
        x_test(numpy.ndarray): input x data for inference
        y_test: input y data for inference

    Returns:
         str: predicted values by the model
    """
    # test set
    x_test = np.array(x_test)
    if y_test is not None:
        y_test = np.array(y_test)
    test_set = {'X': x_test, 'y': y_test}

    mlp_clf = MostCommonWordSense(args.epochs, args.batch_size, None)
    # load existing model
    mlp_clf.load(args.model)

    results = mlp_clf.get_outputs(test_set['X'])

    return results


if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_num_of_senses_to_search', default=3, type=int,
                        action=check_size(0, 100),
                        help='maximum number of senses that are tests')
    parser.add_argument('--input_inference_examples_file',
                        type=validate_existing_filepath,
                        default='data/input_inference_examples.csv',
                        help='input_data_file')
    parser.add_argument('--model', default='data/mcs_model.h5',
                        type=validate_existing_filepath,
                        help='path to the file where the trained model has been stored')
    parser.add_argument('--word_embedding_model_file',
                        type=validate_existing_filepath,
                        default='pretrained_models/GoogleNews-vectors-negative300.bin',
                        help='path to the word embedding\'s model')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs',
                        action=check_size(0, 200))
    parser.add_argument('--batch_size', default=50, type=int, help='batch_size',
                        action=check_size(0, 256))

    args = parser.parse_args()

    # 1. input data
    target_word_vec = read_inference_input_examples_file(args.input_inference_examples_file)
    logger.info("finished reading inference input examples file")

    # 2. Load pre-trained word embeddings model.
    word_embeddings_model = gensim.models.KeyedVectors.\
        load_word2vec_format(args.word_embedding_model_file, binary=True)
    logger.info("finished loading word embeddings model")

    example_cntr = 0
    for input_word in target_word_vec:
        mtype = 'f4, S200'
        sense_data_matrix = np.zeros(0, dtype=mtype)

        i = 0
        # 3. iterate over all synsets of the word
        for synset in wn.synsets(input_word):
            # extract all synset data
            definition, hyps_list, synonym_list = extract_synset_data(synset)
            # 4. feature extraction
            [valid_w2v_flag, definition_sim_cbow, definition_sim, hyps_sim, target_word_emb,
             definition_sentence_emb_cbow] = \
                extract_features_envelope(input_word, definition, hyps_list, word_embeddings_model)

            feature_vec = np.array([definition_sim_cbow, definition_sim, hyps_sim])
            feature_vec = np.concatenate((feature_vec, target_word_emb), 0)
            feature_vec = np.concatenate((feature_vec, definition_sentence_emb_cbow), 0)
            featVecDim = feature_vec.shape[0]
            # X_featureMatrix dim should be (1,featVecDim) but neon classifier gets a minimum of
            # 10 samples not just 1
            X_featureMatrix = np.zeros((10, featVecDim))
            X_featureMatrix[0, :] = feature_vec

            # 5. inference
            classifierOutScore = wsd_classify(x_test=X_featureMatrix, y_test=None)
            data_str = "hyps: " + str(hyps_list[0:2]) + " definition: " + definition
            sense_data_matrix = np.append(
                sense_data_matrix, np.array([(classifierOutScore[0, 1], data_str)], dtype=mtype))

            i = i + 1

            # max num of senses to check
            if i == int(args.max_num_of_senses_to_search):
                break
        example_cntr = example_cntr + 1

        # find sense with max score
        if sense_data_matrix is not None:
            max_val = max(sense_data_matrix,
                          key=lambda sense_data_matrix_entry: sense_data_matrix_entry[0])
            max_val = max_val[0]
            header_text = 'word: ' + input_word
            print(colored(header_text, 'grey', attrs=['bold', 'underline']))

            for data_sense in sense_data_matrix:
                if data_sense[0] == max_val:
                    print(colored(data_sense, 'green', attrs=['bold']))
                else:
                    print(data_sense)

            print()
