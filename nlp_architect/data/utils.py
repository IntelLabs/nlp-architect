# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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
from __future__ import absolute_import, division, print_function

import csv
import os
import random
import sys
from abc import ABC
from io import open
from typing import List, Tuple


class InputExample(ABC):
    """Base class for a single training/dev/test example """

    def __init__(self, guid: str, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence/token classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class Task:
    """ A task definition class
    Args:
        name (str): the name of the task
        processor (DataProcessor): a DataProcessor class containing a dataset loader
        data_dir (str): path to the data source
        task_type (str): the task type (classification/regression/tagging)
    """

    def __init__(self, name: str, processor: DataProcessor, data_dir: str, task_type: str):
        self.name = name
        self.processor = processor
        self.data_dir = data_dir
        self.task_type = task_type

    def get_train_examples(self):
        return self.processor.get_train_examples(self.data_dir)

    def get_dev_examples(self):
        return self.processor.get_dev_examples(self.data_dir)

    def get_test_examples(self):
        return self.processor.get_test_examples(self.data_dir)

    def get_labels(self):
        return self.processor.get_labels()


def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(str(cell, "utf-8") for cell in line)  # noqa: F821
            lines.append(line)
        return lines


def read_column_tagged_file(filename: str, tag_col: int = -1):
    """Reads column tagged (CONLL) style file (tab separated and token per line)
    tag_col is the column number to use as tag of the token (defualts to the last in line)
    return format :
    [ ['token', 'TAG'], ['token', 'TAG2'],... ]
    """
    data = []
    sentence = []
    labels = []
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if len(line) == 0:
                if len(sentence) > 0:
                    data.append((sentence, labels))
                    sentence = []
                    labels = []
                continue
            splits = line.split()
            sentence.append(splits[0])
            labels.append(splits[tag_col])

    if len(sentence) > 0:
        data.append((sentence, labels))
    return data


def write_column_tagged_file(filename: str, data: List[Tuple]):
    file_dir = "{}".format(os.sep).join(filename.split(os.sep)[:-1])
    if not os.path.exists(file_dir):
        raise FileNotFoundError
    with open(filename, "w", encoding="utf-8") as fw:
        for sen in data:
            cols = len(sen)
            items = len(sen[0])
            for i in range(items):
                line = "\t".join([sen[c][i] for c in range(cols)]) + "\n"
                fw.write(line)
            fw.write("\n")


def sample_label_unlabeled(samples: List[InputExample], no_labeled: int, no_unlabeled: int):
    """
    Randomly sample 2 sets of samples from a given collection of InputExamples
    (used for semi-supervised models)
    """
    num_of_examples = len(samples)
    assert no_labeled > 0 and no_unlabeled > 0, "Must provide no_samples > 0"
    assert (
        num_of_examples >= no_labeled + no_unlabeled
    ), "num of total samples smaller than requested sub sets"
    all_indices = list(range(num_of_examples))
    labeled_indices = random.sample(all_indices, no_labeled)
    remaining_indices = list(set(all_indices).difference(set(labeled_indices)))
    unlabeled_indices = random.sample(remaining_indices, no_unlabeled)
    label_samples = [samples[i] for i in labeled_indices]
    unlabel_samples = [samples[i] for i in unlabeled_indices]
    return label_samples, unlabel_samples


def split_column_dataset(
    first_count: int,
    second_count: int,
    out_folder,
    dataset,
    first_filename,
    second_filename,
    tag_col=-1,
):
    """
    Splits a single column tagged dataset into two files according to the amount of examples
    requested to be included in each file.
    split1_count (int) : the amount of examples to include in the first split file
    split2_count (int) : the amount of examples to include in the second split file
    out_folder (str) : the folder in which the result files will be stored
    dataset (str) : the path to the original data file
    split1_filename (str) : the name of the first split file
    split2_filename (str) : the name of the second split file
    tag_col (int) : the index of the tag column
    """
    lines = read_column_tagged_file(dataset, tag_col=tag_col)
    num_of_examples = len(lines)
    assert first_count + second_count <= num_of_examples and first_count > 0 and second_count > 0
    selected_lines = random.sample(lines, first_count + second_count)
    first_data = selected_lines[:first_count]
    second_data = selected_lines[first_count:]
    write_column_tagged_file(out_folder + os.sep + first_filename, first_data)
    write_column_tagged_file(out_folder + os.sep + second_filename, second_data)
