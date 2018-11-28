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

from __future__ import print_function
import numpy as np
import argparse
import json
import os
from tqdm import tqdm
from nlp_architect.utils.io import validate_existing_directory
from nlp_architect.utils.text import SpacyInstance

PAD = "<pad>"
SOS = "<sos>"
UNK = "<unk>"
START_VOCAB = [PAD, SOS, UNK]
tokenizer = SpacyInstance(disable=['tagger', 'ner', 'parser', 'vectors', 'textcat'])


def create_vocabulary(data_list):
    """
    Function to generate vocabulary for both training and development datasets
    """
    vocab = {}
    for list_element in data_list:
        for sentence in list_element:
            for word in sentence:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

    _vocab_list = START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    _vocab_dict = dict((vocab_ele, i) for i, vocab_ele in enumerate(_vocab_list))

    return _vocab_list, _vocab_dict


def create_token_map(para, para_tokens):
    """
    Function to generate mapping between tokens and indices
    """
    token_map = {}
    char_append = ''
    para_token_idx = 0
    for char_idx, char in enumerate(para):
        if char != u' ':
            current_para_token = (para_tokens[para_token_idx])
            char_append += char
            if char_append == current_para_token:
                token_map[char_idx - len(char_append) + 1] = [char_append, para_token_idx]
                para_token_idx += 1
                char_append = ''

    return token_map


def get_glove_matrix(vocabulary_list, download_path):
    """
    Function to obtain preprocessed glove embeddings matrix
    """
    save_file_name = download_path + "glove.trimmed.300"
    if not os.path.exists(save_file_name + ".npz"):
        vocab_len = len(vocabulary_list)
        glove_path = os.path.join(download_path + "glove.6B.300d.txt")
        glove_matrix = np.zeros((vocab_len, 300))
        count = 0
        with open(glove_path) as f:
            for line in tqdm(f):
                split_line = line.lstrip().rstrip().split(" ")
                word = split_line[0]
                word_vec = list(map(float, split_line[1:]))
                if word in vocabulary_list:
                    word_index = vocabulary_list.index(word)
                    glove_matrix[word_index, :] = word_vec
                    count += 1

                if word.upper() in vocabulary_list:
                    word_index = vocabulary_list.index(word.upper())
                    glove_matrix[word_index, :] = word_vec
                    count += 1

                if word.capitalize() in vocabulary_list:
                    word_index = vocabulary_list.index(word.capitalize())
                    glove_matrix[word_index, :] = word_vec
                    count += 1

        print("Saving the embeddings .npz file")
        np.savez_compressed(save_file_name, glove=glove_matrix)
        print("Percentage of words in vocab found in glove embeddings %f" % (count / vocab_len))


# pylint: disable=unnecessary-lambda
def tokenize_sentence(line):
    """
    Function to tokenize  sentence
    """
    tokenized_words = [word.replace("``", '"').replace("''", '"')
                       for word in tokenizer.tokenize(line)]
    return list(map(lambda x: str(x), tokenized_words))


def extract_data_from_files(json_data):
    """
    Function to read and extract data from raw input json files
    """
    data_para = []
    data_ques = []
    data_answer = []
    line_skipped = 0
    for article_id in range(len(json_data['data'])):
        sub_paragraphs = json_data['data'][article_id]['paragraphs']
        for para_id in range(len(sub_paragraphs)):

            req_para = sub_paragraphs[para_id]['context']
            req_para = req_para.replace("''", '" ').replace("``", '" ')
            para_tokens = tokenize_sentence(req_para)
            answer_map = create_token_map(req_para, para_tokens)

            questions = sub_paragraphs[para_id]['qas']
            for ques_id in range(len(questions)):

                req_question = questions[ques_id]['question']
                question_tokens = tokenize_sentence(req_question)

                for ans_id in range(1):
                    answer_text = questions[ques_id]['answers'][ans_id]['text']
                    answer_start = questions[ques_id][
                        'answers'][ans_id]['answer_start']
                    answer_end = answer_start + len(answer_text)
                    text_tokens = tokenize_sentence(answer_text)
                    last_word_answer = len(text_tokens[-1])
                    try:
                        a_start_idx = answer_map[answer_start][1]

                        a_end_idx = answer_map[
                            answer_end - last_word_answer][1]

                        data_para.append(para_tokens)
                        data_ques.append(question_tokens)
                        data_answer.append((a_start_idx, a_end_idx))

                    except KeyError:
                        line_skipped += 1

    return data_para, data_ques, data_answer


def write_to_file(file_dict, path_to_save):
    """
    Function to write data to files
    """

    for f_name in file_dict:
        if f_name == "vocab.dat":
            with open(os.path.join(path_to_save + f_name), 'w') as target_file:
                for word in file_dict[f_name]:
                    target_file.write(str(word) + "\n")
        else:
            with open(os.path.join(path_to_save + f_name), 'w') as target_file:
                for line in file_dict[f_name]:
                    target_file.write(" ".join([str(tok) for tok in line]) + "\n")


def get_ids_list(data_list, vocab):
    """
    Function to obtain indices from vocabulary
    """
    ids_list = []
    for line in data_list:
        curr_line_idx = []
        for word in line:
            try:
                curr_line_idx.append(vocab[word])
            except ValueError:
                curr_line_idx.append(vocab[UNK])

        ids_list.append(curr_line_idx)
    return ids_list


if __name__ == '__main__':

    # parse the command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', help='enter path where training data and the \
                        glove embeddings were downloaded',
                        type=validate_existing_directory)

    parser.add_argument('--no_preprocess_glove', action="store_true",
                        help='Chose whether or not to preprocess glove embeddings')

    parser.set_defaults()
    args = parser.parse_args()

    glove_flag = not args.no_preprocess_glove

    # Validate files in the folder:
    missing_flag = 0
    files_list = ["train-v1.1.json", "dev-v1.1.json"]
    for file_name in files_list:
        if not os.path.exists(os.path.join(args.data_path, file_name)):
            print("The following required file is missing :", file_name)
            missing_flag = 1

    if missing_flag:
        print("Please ensure that required datasets are downloaded")
        exit()

    data_path = args.data_path
    # Load Train and Dev Data
    train_filename = os.path.join(data_path, "train-v1.1.json")
    dev_filename = os.path.join(data_path, "dev-v1.1.json")
    with open(train_filename) as train_file:
        train_data = json.load(train_file)

    with open(dev_filename) as dev_file:
        dev_data = json.load(dev_file)

    print('Extracting data from json files')
    # Extract training data from raw files
    train_para, train_question, train_ans = extract_data_from_files(train_data)
    # Extract dev data from raw dataset
    dev_para, dev_question, dev_ans = extract_data_from_files(dev_data)

    data_lists = [train_para, train_question, dev_para, dev_question]
    # Obtain vocab list
    print("Creating Vocabulary")
    vocab_list, vocab_dict = create_vocabulary(data_lists)

    # Obtain embedding matrix from pre-trained glove vectors:
    if glove_flag:
        print("Preprocessing glove")
        get_glove_matrix(vocab_list, data_path)

    # Get vocab ids file
    train_para_ids = get_ids_list(train_para, vocab_dict)
    train_question_ids = get_ids_list(train_question, vocab_dict)
    dev_para_ids = get_ids_list(dev_para, vocab_dict)
    dev_question_ids = get_ids_list(dev_question, vocab_dict)

    final_data_dict = {"train.ids.context": train_para_ids,
                       "train.ids.question": train_question_ids,
                       "dev.ids.context": dev_para_ids,
                       "dev.ids.question": dev_question_ids,
                       "vocab.dat": vocab_list,
                       "train.span": train_ans,
                       "dev.span": dev_ans}

    print("writing data to files")
    write_to_file(final_data_dict, data_path)
