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

import logging
import os

from nlp_architect.data.utils import DataProcessor
from nlp_architect.utils.text import Vocabulary
from nlp_architect.utils.utils_squad import read_squad_examples


logger = logging.getLogger(__name__)


class QuestionAnsweringProcessor(DataProcessor):
    """Question Answering Processor dataset loader.
    Loads a directory with train/dev files.
    """

    def __init__(self, data_dir, version_2_with_negative):
        if not os.path.exists(data_dir):
            raise FileNotFoundError
        self.data_dir = data_dir
        self.version_2_with_negative = version_2_with_negative

    def _read_examples(self, file_name, set_name):
        if not os.path.exists(self.data_dir + os.sep + file_name):
            logger.error(
                "Requested file %s in path %s for TokenClsProcess not found",
                file_name, self.data_dir)
            return None
        return read_squad_examples(
            os.path.join(self.data_dir, file_name),
            is_training=(set_name == 'train'),
            version_2_with_negative=self. version_2_with_negative
        )

    def get_train_examples(self):
        file_name = "train-v2.0.json" if self.version_2_with_negative else "train-v1.1.json"
        return self._read_examples(file_name, "train")

    def get_dev_examples(self):
        file_name = "dev-v2.0.json" if self.version_2_with_negative else "dev-v1.1.json"
        return self._read_examples(file_name, "dev")

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_vocabulary(self):
        """See base class."""
        examples = self.get_train_examples() + self.get_dev_examples()
        vocab = Vocabulary(start=1)
        for ex in examples:
            for tok in ex.tokens:
                vocab.add(tok)
        return vocab
