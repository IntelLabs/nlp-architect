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
import ngraph as ng
from collections import Counter


def max_values_squad(data_train):
    """
    Function to compute the maximum length of sentences in
    paragraphs and questions

    Arguments:
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


def get_embeddding_array(embedding_size):
    """
    Function to compute the maximum length of sentences in
    paragraphs and questions

    Arguments:
    -----------
    data_train: list containing the entire dataset
    Returns:
    --------
    maximum length of question and paragraph
    """

    embed_path = "data/squad/glove.trimmed.{}.npz".format(embedding_size)
    embeddingz = np.load(embed_path)
    embeddings = embeddingz['glove']
    max_vocab = embeddings.shape[0]
    return max_vocab, embeddings


def create_squad_training(
        paras_file,
        ques_file,
        answer_file,
        vocab_file,
        data_train_len=None):
    """
    Function to read data from preprocessed files and return lists

    Arguments:
    -----------
    paras_file: File name for preprocessed paragraphs
    ques_file: File name for preprocessed questions
    answer_file: File name for preprocessed answer spans
    vocab_file:  File name for preprocessed vocab
    data_train_len= length of train dataset to use

    Return Value:
    --------
    append list for train/dev dataset
    list for vocab

    """
    para_list = []
    ques_list = []
    ans_list = []
    vocab_list = []

    with open(paras_file) as f_para:
        for ele in f_para:
            para_list.append(list(map(int, ele.strip().split())))

    with open(ques_file) as f_ques:
        for ele in f_ques:
            ques_list.append(list(map(int, ele.strip().split())))

    with open(answer_file) as f_ans:
        for ele in f_ans:
            ans_list.append(list(map(int, ele.strip().split())))

    with open(vocab_file) as f_vocab:
        for ele in f_vocab:
            vocab_list.append(ele.strip().split())

    data_train = []
    if data_train_len is None:
        data_train_len = len(para_list)

    for idx in range(data_train_len):
        data_train.append([para_list[idx], ques_list[idx], ans_list[idx]])

    return data_train, vocab_list


def get_data_array_squad_ngraph(
        params_dict,
        data_train,
        set_val='train',
        vocab_train=None):
    """
    Function to return a dictionary for train/dev datasets in the format required
    by ArrayIterator object in ngraph
    Arguments:
    ---------
    params_dict: dictionary containing all input parameters
    data_train: list containing the training data_train
    set_val: indicates id its a training set or dev set

    Return_Value:
    ----------
    Returns a dictionary in the format required by ArrayIterator object

    """
    train=create_train_dict()
    max_para = params_dict['max_para']
    max_question = params_dict['max_question']
    question_mask = np.zeros([1, max_question])
    para_mask = np.zeros([1, max_para])

    count = 0
    for ele in data_train:

        para = ele[0]
        para_idx = ele[0]

        if len(para) < max_para:
            #import ipdb;ipdb.set_trace()
            pad_length = max_para - len(para)
            para_idx = para_idx + [0] * pad_length
            para_len = np.zeros([1, max_para])
            para_mask = np.zeros([1, max_para])
            para_len[0, 0:len(para)] = 1
            para_mask[0, len(para) - 1] = 1

            question = ele[1]
            question_idx = ele[1]

            if len(question) < max_question:
                # Pad to the right to max length of question
                pad_length = max_question - len(question)
                question_idx = question_idx + [0] * pad_length
                question_len = np.zeros([1, max_question])
                question_mask = np.zeros([1, max_question])
                question_len[0, 0:len(question)] = 1
                question_mask[0, len(question) - 1] = 1

            train['para']['data'].append(para_idx)
            train['answer']['data'].append(ele[2])
            train['question']['data'].append(question_idx)
            train['question_len']['data'].append(question_len)
            train['para_len']['data'].append(para_len)
            train['para_mask']['data'].append(para_mask)
            train['question_mask']['data'].append(question_mask)

            if set_val == 'train':
                train['dropout_val']['data'].append(0.6)

            else:
                train['dropout_val']['data'].append(1.0)

            if set_val == 'train':
                if count >= params_dict['train_set_size']:
                    break
                count += 1

            else:
                if count >= 2000:
                    break
                count += 1

    train_out=get_output_dict(train,max_question)

    return train_out


def create_train_dict():
    """
    Function to define the data dictionary with format as required by
    ArrayIterator object

    """
    data = {}
    data['para'] = {}
    data['answer'] = {}
    data['question'] = {}
    data['question_len'] = {}
    data['para_len'] = {}
    data['question_mask'] = {}
    data['para_mask'] = {}
    data['dropout_val'] = {}

    data['para']['data'] = []
    data['para']['axes'] = ()
    data['question']['data'] = []
    data['question']['axes'] = ()

    data['question_len']['data'] = []
    data['question_len']['axes'] = ()
    data['para_len']['data'] = []
    data['para_len']['axes'] = ()

    data['question_mask']['data'] = []
    data['question_mask']['axes'] = ()
    data['para_mask']['data'] = []
    data['para_mask']['axes'] = ()

    data['answer']['data'] = []
    data['answer']['axes'] = ()
    data['dropout_val']['data'] = []
    data['dropout_val']['axes'] = ()

    return data

def get_output_dict(train,max_question):
    """
    Function to populate data dictionary with data and defined axes as
    required by ArrayIterator object in ngraph
    """

    train['para']['data'] = np.array(
        [xi for xi in train['para']['data'][:-1]], dtype=np.int32)
    train['question']['data'] = np.array(
        [xi for xi in train['question']['data'][:-1]], dtype=np.int32)
    train['para_len']['data'] = np.array(
        [xi for xi in train['para_len']['data'][:-1]], dtype=np.int32)
    train['question_len']['data'] = np.array(
        [xi for xi in train['question_len']['data'][:-1]], dtype=np.int32)
    train['question_mask']['data'] = np.array(
        [xi for xi in train['question_mask']['data'][:-1]], dtype=np.int32)
    train['para_mask']['data'] = np.array(
        [xi for xi in train['para_mask']['data'][:-1]], dtype=np.int32)

    train['answer']['data'] = np.array(
        train['answer']['data'][:-1], dtype=np.int32)
    train['dropout_val']['data'] = np.array(
        train['dropout_val']['data'][:-1], dtype=np.float32)

    REC2 = ng.make_axis(length=max_question, name='REC2')

    span = ng.make_axis(length=2, name='span')
    dummy_axis = ng.make_axis(length=1, name='dummy_axis')
    train['para']['axes'] = ('batch', 'REC')
    train['question']['axes'] = ('batch', 'REC2')
    train['para_len']['axes'] = ('batch', 'dummy_axis', 'REC')
    train['question_len']['axes'] = ('batch', 'dummy_axis', 'REC2')
    train['answer']['axes'] = ('batch', 'span')
    train['question_mask']['axes'] = ('batch', 'dummy_axis', 'REC2')
    train['para_mask']['axes'] = ('batch', 'dummy_axis', 'REC')
    train['dropout_val']['axes'] = ('batch')

    return train

def cal_f1_score(params_dict, ground_truths, predictions):
    """
    Function to calculate F-1 and EM scores given predictions and ground truths
    """
    preds1 = np.transpose(predictions[:, 0, :])
    preds2 = np.transpose(predictions[:, 1, :])

    start_idx, end_idx = obtain_indices(preds1, preds2)

    f1 = 0
    exact_match = 0
    for i in range(params_dict['batch_size']):
        ele1 = start_idx[i]
        ele2 = end_idx[i]
        preds = np.linspace(ele1, ele2, abs(ele2 - ele1 + 1))
        length_gts = abs(ground_truths[i][1] - ground_truths[i][0] + 1)
        gts = np.linspace(ground_truths[i][0], ground_truths[i][1], length_gts)
        common = Counter(preds) & Counter(gts)
        num_same = sum(common.values())

        exact_match += int(np.array_equal(preds, gts))

        if num_same == 0:
            f1 += 0
        else:
            assert(len(preds)>0 and len(gts)>0)

            precision = 1.0 * num_same / len(preds)
            recall = 1.0 * num_same / len(gts)
            f1 += (2 * precision * recall) / (precision + recall)

    return 100 * (f1 / params_dict['batch_size']), 100 * \
        (exact_match / params_dict['batch_size'])


def obtain_indices(preds_start,preds_end):
    """
    Function to get answer indices given the predictions
    """
    ans_start=[]
    ans_end=[]
    for i in range(preds_start.shape[0]):
        max_ans_id = -100000000
        st_idx=0
        en_idx=0
        ele1=preds_start[i]
        ele2=preds_end[i]
        len_para = len(ele1)
        for j in range(len_para):
            for k in range(15):
                if j + k >= len_para:
                    break
                ans_start_int = ele1[j]
                ans_end_int = ele2[j + k]
                if (ans_start_int + ans_end_int) > max_ans_id:
                    max_ans_id = ans_start_int + ans_end_int
                    st_idx = j
                    en_idx=j+k

        ans_start.append(st_idx)
        ans_end.append(en_idx)

    return (np.array(ans_start), np.array(ans_end))
