# -*- coding: utf-8 -*-
# file: main.py
# author: JinTian
# time: 11/03/2017 9:53 AM
# Copyright 2017 JinTian. All Rights Reserved.
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
# ------------------------------------------------------------------------
import tensorflow as tf
from gen_char_rnn import rnn_model
from data_process import process_datas
import numpy as np

start_token = 'B'
end_token = 'E'
model_dir = '../../nlp_architect/api/gen-pretrained'
corpus_file = '../../datasets/gen_data/gongchengche_2018_10_08.csv'

lr = 0.0002


def to_word(predict, vocabs):
    predict = predict[0]       
    predict /= np.sum(predict)
    sample = np.random.choice(np.arange(len(predict)), p=predict)
    if sample > len(vocabs):
        return vocabs[-1]
    else:
        return vocabs[sample]


def gen_data(begin_word):
    batch_size = 1
    print('## loading corpus from %s' % model_dir)
    datas_vector, word_int_map, vocabularies = process_datas(corpus_file)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=lr)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, checkpoint)

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        if begin_word:
            word = begin_word
        else:
            word = to_word(predict, vocabularies)
        data_ = ''

        i = 0
        while word != end_token:
            data_ += word
            i += 1
            if i >= 100:
                break
            x = np.zeros((1, 1))
            x[0, 0] = word_int_map[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)

        return data_


def pretty_print_data(data_):
    data_sentences = data_.split('。')
    for s in data_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')

if __name__ == '__main__':
    begin_char = input('## please input the first character:')
    data = gen_data(begin_char)
    pretty_print_data(data_=data)