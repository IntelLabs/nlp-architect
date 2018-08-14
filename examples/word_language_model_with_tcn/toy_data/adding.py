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
"""
Data generation and data loader for the synthetic adding dataset
"""
import numpy as np


class Adding:
    """
    Iterator class that generates data and provides batches for training
    """
    def __init__(self, seq_len=200, n_train=50000, n_test=1000, batch_size=32):
        """
        Initialize class
        Args:
            seq_len: int, sequence length of data
            n_train: int, number of training samples
            n_test: int, number of test samples
            batch_size: int, number of samples per batch
        """
        self.seq_len = seq_len
        self.n_train = n_train
        self.n_test = n_test

        x_train, y_train = self.load_data(n_train)
        x_val, y_val = self.load_data(n_test)

        self.train = (x_train, y_train)

        self.test = (x_val, y_val)

        self.sample_count = 0
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def get_batch(self):
        """
        Return one batch at a time
        Returns:
            Tuple of two numpy arrays, first is input data sequence, second is the addition output
        """
        if self.sample_count + self.batch_size > self.n_train:
            self.sample_count = 0

        batch = (self.train[0][self.sample_count: self.sample_count + self.batch_size],
                 self.train[1][self.sample_count: self.sample_count + self.batch_size])
        self.sample_count += self.batch_size
        return batch

    def load_data(self, num_samples):
        """
        Generate and load the data into numpy arrays
        Args:
            num_samples: # of data in the set
        Returns:
            Tuple of two numpy arrays, first is input data sequence, second is the addition output
        """
        x_num = np.random.rand(num_samples, self.seq_len, 1)
        x_mask = np.zeros((num_samples, self.seq_len, 1))
        y_data = np.zeros((num_samples, 1))
        for i in range(num_samples):
            positions = np.random.choice(self.seq_len, size=2, replace=False)
            x_mask[i, positions[0], 0] = 1
            x_mask[i, positions[1], 0] = 1
            y_data[i, 0] = x_num[i, positions[0], 0] + x_num[i, positions[1], 0]
        x_data = np.concatenate((x_num, x_mask), axis=2)
        return x_data, y_data
