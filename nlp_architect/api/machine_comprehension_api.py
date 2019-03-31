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

from __future__ import division
from __future__ import print_function

import os
import re
import zipfile
from os import makedirs
from random import shuffle

import numpy as np
import tensorflow as tf

from nlp_architect.api.abstract_api import AbstractApi
from nlp_architect.models.matchlstm_ansptr import MatchLSTMAnswerPointer
from nlp_architect import LIBRARY_OUT
from nlp_architect.utils.generic import license_prompt
from nlp_architect.utils.io import download_unlicensed_file
from nlp_architect.utils.mrc_utils import (
    create_squad_training, max_values_squad, get_data_array_squad)


class MachineComprehensionApi(AbstractApi):
    """
    Machine Comprehension API
    """
    dir = str(LIBRARY_OUT / 'mrc-pretrained')
    data_path = os.path.join(dir, 'mrc_data', 'data')
    data_dir = os.path.join(dir, 'mrc_data')
    model_dir = os.path.join(dir, 'mrc_trained_model')
    model_path = os.path.join(dir, 'mrc_trained_model', 'trained_model')

    def __init__(self, prompt=True):
        self.prompt = None
        self.vocab_dict = None
        self.vocab_rev = None
        self.model = None
        self.dev = None
        self.sess = None
        self.prompt = prompt
        self.params_dict = {'batch_size': 1,
                            'hidden_size': 150,
                            'max_para': 300,
                            'epoch_no': 15,
                            'inference_only': True}
        self.file_name_dict = {'train_para_ids': 'train.ids.context',
                               'train_ques_ids': 'train.ids.question',
                               'train_answer': 'train.span',
                               'val_para_ids': 'dev.ids.context',
                               'val_ques_ids': 'dev.ids.question',
                               'val_ans': 'dev.span',
                               'vocab_file': 'vocab.dat',
                               'embedding': 'glove.trimmed.300.npz'}

    def download_model(self):
        # Validate contents of data_path folder:
        data_path = self.data_path
        download = False
        for file_name in self.file_name_dict.values():
            if not os.path.exists(os.path.join(data_path, file_name)):
                # prompt
                download = True
                print("The following required file is missing :", file_name)

        if download is True:
            if self.prompt is True:
                license_prompt('mrc_data',
                               'https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/mrc'
                               '/mrc_data.zip',
                               self.data_dir)
                license_prompt('mrc_model',
                               'https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/mrc'
                               '/mrc_model.zip',
                               self.model_dir)
            data_zipfile = os.path.join(self.data_dir, 'mrc_data.zip')
            model_zipfile = os.path.join(self.model_dir, 'mrc_model.zip')
            makedirs(self.data_dir, exist_ok=True)
            makedirs(self.model_dir, exist_ok=True)
            download_unlicensed_file('https://s3-us-west-2.amazonaws.com/nlp-architect-data'
                                     '/models/mrc/',
                                     'mrc_data.zip', data_zipfile)
            download_unlicensed_file('https://s3-us-west-2.amazonaws.com/nlp-architect-data'
                                     '/models/mrc/',
                                     'mrc_model.zip', model_zipfile)
            with zipfile.ZipFile(data_zipfile) as data_zip_ref:
                data_zip_ref.extractall(self.data_dir)
            with zipfile.ZipFile(model_zipfile) as model_zip_ref:
                model_zip_ref.extractall(self.model_dir)

    def load_model(self):
        select_device = 'GPU'
        restore_model = True
        # Create dictionary of filenames
        self.download_model()

        data_path = self.data_path
        # Paths for preprcessed files
        path_gen = data_path  # data is actually in mrc_data/data not, mrc_data
        train_para_ids = os.path.join(path_gen, self.file_name_dict['train_para_ids'])
        train_ques_ids = os.path.join(path_gen, self.file_name_dict['train_ques_ids'])
        answer_file = os.path.join(path_gen, self.file_name_dict['train_answer'])
        val_paras_ids = os.path.join(path_gen, self.file_name_dict['val_para_ids'])
        val_ques_ids = os.path.join(path_gen, self.file_name_dict['val_ques_ids'])
        val_ans_file = os.path.join(path_gen, self.file_name_dict['val_ans'])
        vocab_file = os.path.join(path_gen, self.file_name_dict['vocab_file'])

        model_dir = self.model_path
        # Create model dir if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = model_dir

        # Create lists for train and validation sets
        data_train = create_squad_training(train_para_ids, train_ques_ids, answer_file)
        data_dev = create_squad_training(val_paras_ids, val_ques_ids, val_ans_file)
        with open(vocab_file, encoding='UTF-8') as fp:
            vocab_list = fp.readlines()
        self.vocab_dict = {}
        self.vocab_rev = {}

        for i in range(len(vocab_list)):
            self.vocab_dict[i] = vocab_list[i].strip()
            self.vocab_rev[vocab_list[i].strip()] = i

            self.params_dict['train_set_size'] = len(data_train)

        # Combine train and dev data
        data_total = data_train + data_dev

        # obtain maximum length of question
        _, max_question = max_values_squad(data_total)
        self.params_dict['max_question'] = max_question

        # Load embeddings for vocab
        print('Loading Embeddings')
        embeddingz = np.load(os.path.join(path_gen, self.file_name_dict['embedding']))
        embeddings = embeddingz['glove']

        # Create train and dev sets
        print("Creating training and development sets")
        self.dev = get_data_array_squad(self.params_dict, data_dev, set_val='val')

        # Define Reading Comprehension model
        with tf.device('/device:' + select_device + ':0'):
            self.model = MatchLSTMAnswerPointer(self.params_dict, embeddings)

        # Define Configs for training
        run_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

        # Create session run training
        self.sess = tf.Session(config=run_config)
        init = tf.global_variables_initializer()

        # Model Saver
        # pylint: disable=no-member
        model_saver = tf.train.Saver()
        model_ckpt = tf.train.get_checkpoint_state(model_path)
        idx_path = model_ckpt.model_checkpoint_path + ".index" if model_ckpt else ""

        # Initialize with random or pretrained weights
        # pylint: disable=no-member
        if model_ckpt and restore_model and (tf.gfile.Exists(
                model_ckpt.model_checkpoint_path) or tf.gfile.Exists(idx_path)):
            model_saver.restore(self.sess, model_ckpt.model_checkpoint_path)
            print("Loading from previously stored session")
        else:
            self.sess.run(init)

        shuffle(self.dev)

    @staticmethod
    def paragraphs(valid, vocab_tuple, num_examples):
        paragraphs = []
        vocab_forward = vocab_tuple[0]
        for idx in range(num_examples):
            test_paragraph = [vocab_forward[ele] for ele in valid[idx][0] if ele != 0]
            para_string = " ".join(map(str, test_paragraph))
            paragraphs.append(re.sub(r'\s([?.!,"](?:\s|$))', r'\1', para_string))  # (?:\s|$))
        return paragraphs

    @staticmethod
    def questions(valid, vocab_tuple, num_examples):
        vocab_forward = vocab_tuple[0]
        questions = []
        for idx in range(num_examples):
            test_question = [vocab_forward[ele] for ele in valid[idx][1] if ele != 0]
            ques_string = " ".join(map(str, test_question))
            questions.append(re.sub(r'\s([?.!"",])', r'\1', ques_string))
        return questions

    def inference(self, doc):
        body = doc
        print("Begin Inference Mode")
        question = body['question']
        paragraph_id = body['paragraph']
        return self.model.inference_mode(self.sess, self.dev, [self.vocab_dict, self.vocab_rev],
                                         dynamic_question_mode=True, num_examples=1, dropout=1.0,
                                         dynamic_usr_question=question,
                                         dynamic_question_index=paragraph_id)

    def get_paragraphs(self):
        ret = {'paragraphs': self.paragraphs(self.dev, [self.vocab_dict, self.vocab_rev],
                                             num_examples=5),
               'questions': self.questions(self.dev, [self.vocab_dict, self.vocab_rev],
                                           num_examples=5)}
        return ret
