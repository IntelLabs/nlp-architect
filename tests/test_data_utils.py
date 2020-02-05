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

import math
import os
from nlp_architect.data.utils import split_column_dataset
from tests.utils import count_examples


def test_split_dataset():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, "fixtures/data/distillation")
    num_of_examples = count_examples(data_dir + os.sep + "train.txt")
    labeled_precentage = 0.4
    unlabeled_precentage = 0.5
    if os.path.exists(data_dir):
        labeled_file = "labeled.txt"
        unlabeled_file = "unlabeled.txt"
        split_column_dataset(
            dataset=os.path.join(data_dir, "train.txt"),
            first_count=math.ceil(num_of_examples * labeled_precentage),
            second_count=math.ceil(num_of_examples * unlabeled_precentage),
            out_folder=data_dir,
            first_filename=labeled_file,
            second_filename=unlabeled_file,
        )
        check_labeled_count = count_examples(data_dir + os.sep + labeled_file)
        assert check_labeled_count == math.ceil(num_of_examples * labeled_precentage)
        check_unlabeled_count = count_examples(data_dir + os.sep + unlabeled_file)
        assert check_unlabeled_count == math.ceil(num_of_examples * unlabeled_precentage)
        os.remove(data_dir + os.sep + "labeled.txt")
        os.remove(data_dir + os.sep + "unlabeled.txt")
