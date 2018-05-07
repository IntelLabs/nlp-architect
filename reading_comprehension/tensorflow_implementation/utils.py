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
from collections import Counter
import os
from tqdm import tqdm


def max_values_squad(data_train):
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


def build_vocab_snli(data_train):
    vocab = {}
    idx = 1
    for ele in data_train:
        for word in ele[0]:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
        for word in ele[1]:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


def get_embeddding_array(embedding_size):

    embed_path = "data/squad/glove.trimmed.{}.npz".format(embedding_size)
    embeddingz = np.load(embed_path)
    embeddings = embeddingz['glove']
    max_vocab = embeddings.shape[0]
    return max_vocab, embeddings


def get_qids(args, q_id_path, data_dev):
    q_ids = open(q_id_path)
    qids_list = []
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


def create_squad_training(
        paras_file,
        ques_file,
        answer_file,
        data_train_len=None):
    f_para = open(paras_file)
    f_ques = open(ques_file)
    f_ans = open(answer_file)
    para_list = []
    ques_list = []
    ans_list = []
    for ele in f_para:
        para_list.append(list(map(int, ele.strip().split())))
    for ele in f_ques:
        ques_list.append(list(map(int, ele.strip().split())))

    for ele in f_ans:
        ans_list.append(list(map(int, ele.strip().split())))

    data_train = []
    if data_train_len is None:
        data_train_len = len(para_list)

    for idx in range(data_train_len):
        data_train.append([para_list[idx], ques_list[idx], ans_list[idx]])

    return data_train


def get_data_array_squad(params_dict, data_train, set_val='train'):

    max_para = params_dict['max_para']
    max_question = params_dict['max_question']

    question_mask = np.zeros([1, max_question])
    para_mask = np.zeros([1, max_para])
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

            train_set.append(
                (para_idx,
                 question_idx,
                 para_len,
                 question_len,
                 ele[2],
                    para_mask,
                    ques_mask))

            if set_val == 'train':
                count += 1
                if count >=params_dict['train_set_size']:
                    break

    print (len(train_set))
    return train_set


def cal_f1_score(batch_size, ground_truths, predictions):
    """
    Function to calculate F-1 and EM scores given predictions and ground truths
    """

    start_idx, end_idx = obtain_indices(predictions[0], predictions[1])
    f1 = 0
    exact_match = 0
    for i in range(batch_size):
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
            precision = 1.0 * num_same / len(preds)
            recall = 1.0 * num_same / len(gts)
            f1 += (2 * precision * recall) / (precision + recall)
    import ipdb
    ipdb.set_trace
    return 100 * (f1 / batch_size), 100 * (exact_match / batch_size)


def create_data_dict(data):

    train = {}
    train['para'] = []
    train['answer'] = []
    train['question'] = []
    train['question_len'] = []
    train['para_len'] = []
    train['para_mask'] = []
    train['question_mask'] = []

    for (
        para_idx,
        question_idx,
        para_len,
        question_len,
        answer,
        para_mask,
            ques_mask) in data:

        train['para'].append(para_idx)
        train['question'].append(question_idx)
        train['para_len'].append(para_len)
        train['question_len'].append(question_len)
        train['answer'].append(answer)
        train['para_mask'].append(para_mask)
        train['question_mask'].append(ques_mask)

    return train


def obtain_indices(preds_start, preds_end):
    """
    Function to get answer indices given the predictions
    """
    ans_start = []
    ans_end = []
    for i in range(preds_start.shape[0]):
        max_ans_id = -100000000
        st_idx = 0
        en_idx = 0
        ele1 = preds_start[i]
        ele2 = preds_end[i]
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
                    en_idx = j + k

        ans_start.append(st_idx)
        ans_end.append(en_idx)

    return (np.array(ans_start), np.array(ans_end))
