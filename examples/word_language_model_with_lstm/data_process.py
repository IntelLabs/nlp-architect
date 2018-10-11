# -*- coding: utf-8 -*-
# file: datas.py
# author: JinTian
# time: 08/03/2017 7:39 PM
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
import collections
import os
import sys
import numpy as np

start_token = 'B'
end_token = 'E'


def process_datas(file_name):
    # datas -> list of numbers
    datas = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                content = line.strip()
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 75:
                    continue
                content = start_token + content + end_token
                datas.append(content)
            except ValueError as e:
                pass
    # datas = sorted(datas, key=len)

    all_words = [word for poem in datas for word in poem]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    words, _ = zip(*count_pairs)

    words = words + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    datas_vector = [list(map(lambda word: word_int_map.get(word, len(words)), data)) for data in datas]

    return datas_vector, word_int_map, words


def generate_batch(batch_size, datas_vec, word_to_int):
    n_chunk = len(datas_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = datas_vec[start_index:end_index]
        length = max(map(len, batches))
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row, batch in enumerate(batches):
            x_data[row, :len(batch)] = batch
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches