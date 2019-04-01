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

import numpy as np


def max_values_squad(data_train):

    """
    Function to compute the maximum length of sentences in
    paragraphs and questions

    Args:
    -----------
    data_train: list containing the entire dataset

    Returns:
    --------
    maximum length of question and paragraph
    """
    max_hypothesis = len(data_train[0][1])
    max_premise = len(data_train[0][0])

    for ele in data_train:
        len1 = len(ele[0])
        len2 = len(ele[1])
        if max_premise < len1:
            max_premise = len1
        if max_hypothesis < len2:
            max_hypothesis = len2

    return max_premise, max_hypothesis


def get_qids(args, q_id_path, data_dev):

    """
    Function to create a list of question_ids in dev set

    Args:
    -----------
    q_id_path: path to question_ids file
    data_dev: development set

    Returns:
    --------
    list of question ids
    """
    qids_list = []
    with open(q_id_path) as q_ids:
        for ele in q_ids:
            qids_list.append(ele.strip().replace(" ", ""))

    final_qidlist = []
    count = 0
    for ele in data_dev:
        para_idx = ele[0]
        if len(para_idx) < args.max_para:
            final_qidlist.append(qids_list[count])
        count += 1

    return final_qidlist


def create_squad_training(paras_file, ques_file, answer_file, data_train_len=None):

    """
    Function to read data from preprocessed files and return
    data in the form of a list

    Args:
    -----------
    paras_file: File name for preprocessed paragraphs
    ques_file: File name for preprocessed questions
    answer_file: File name for preprocessed answer spans
    vocab_file:  File name for preprocessed vocab
    data_train_len= length of train dataset to use

    Returns:
    --------
    appended list for train/dev dataset
    """

    para_list = []
    ques_list = []
    ans_list = []
    with open(paras_file) as f_para:
        for ele in f_para:
            para_list.append(list(map(int, ele.strip().split())))

    with open(ques_file) as f_ques:
        for ele in f_ques:
            ques_list.append(list(map(int, ele.strip().split())))

    with open(answer_file) as f_ans:
        for ele in f_ans:
            ans_list.append(list(map(int, ele.strip().split())))

    data_train = []
    if data_train_len is None:
        data_train_len = len(para_list)

    for idx in range(data_train_len):
        data_train.append([para_list[idx], ques_list[idx], ans_list[idx]])

    return data_train


def get_data_array_squad(params_dict, data_train, set_val='train'):
    """
    Function to pad all sentences and restrict to max length defined by user

    Args:
    ---------
    params_dict: dictionary containing all input parameters
    data_train: list containing the training/dev data_train
    set_val: indicates id its a training set or dev set

    Returns:
    ----------
    Returns a list of tuples with padded sentences and masks
    """

    max_para = params_dict['max_para']
    max_question = params_dict['max_question']
    train_set = []
    count = 0
    for ele in data_train:

        para = ele[0]
        para_idx = ele[0]

        if len(para_idx) < max_para:
            pad_length = max_para - len(para_idx)
            para_idx = para_idx + [0] * pad_length
            para_len = len(para)
            para_mask = np.zeros([1, max_para])
            para_mask[0, 0:len(para)] = 1
            para_mask = para_mask.tolist()[0]

            question_idx = ele[1]
            question = ele[1]

            if len(question) < max_question:
                pad_length = max_question - len(question)
                question_idx = question_idx + [0] * pad_length
                question_len = len(question)
                ques_mask = np.zeros([1, max_question])
                ques_mask[0, 0:question_len] = 1
                ques_mask = ques_mask.tolist()[0]

            train_set.append((para_idx, question_idx, para_len, question_len,
                              ele[2], para_mask, ques_mask))

            if set_val == 'train':
                count += 1
                if count >= params_dict['train_set_size']:
                    break
    return train_set


def create_data_dict(data):

    """
    Function to convert data to dictionary format

    Args:
    -----------
    data: train/dev data as a list

    Returns:
    --------
    a dictionary containing dev/train data
    """

    train = {}
    train['para'] = []
    train['answer'] = []
    train['question'] = []
    train['question_len'] = []
    train['para_len'] = []
    train['para_mask'] = []
    train['question_mask'] = []

    for (para_idx, question_idx, para_len, question_len, answer, para_mask, ques_mask) in data:

        train['para'].append(para_idx)
        train['question'].append(question_idx)
        train['para_len'].append(para_len)
        train['question_len'].append(question_len)
        train['answer'].append(answer)
        train['para_mask'].append(para_mask)
        train['question_mask'].append(ques_mask)

    return train
