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
import json
from nlp_architect.data.utils import DataProcessor
from nlp_architect.utils.text import Vocabulary
logger = logging.getLogger(__name__)


class QuestionAnsweringInputExample():

    """A single training/test example for question answering."""

    def __init__(
            self, qas_id: str, question_text, doc_tokens, orig_answer_text, 
            start_position, end_position, is_impossible):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class QuestionAnsweringProcessor(DataProcessor):
    """Question Answering Processor dataset loader.
    Loads a directory with train/dev files.
    """

    def __init__(self, data_dir, version_2_with_negative):
        if not os.path.exists(data_dir):
            raise FileNotFoundError
        self.data_dir = data_dir
        self.version_2_with_negative = version_2_with_negative

    def _read_examples(self, data_dir, file_name, set_name):
        if not os.path.exists(data_dir + os.sep + file_name):
            logger.error("Requested file {} in path {} for TokenClsProcess not found".format(
                file_name, data_dir))
            return None
        return read_squad_examples(
            os.path.join(data_dir, file_name),
            is_training=(set_name == 'train'),
            version_2_with_negative = self. version_2_with_negative
            )

    def get_train_examples(self):
        return self._read_examples(self.data_dir, "train-v1.1.json", "train")

    def get_dev_examples(self):
        return self._read_examples(self.data_dir, "dev-v1.1.json", "dev")

    def get_test_examples(self):
        return self._read_examples(self.data_dir, "test-v1.1.json", "test")

    # # pylint: disable=arguments-differ
    # def get_labels(self):
    #     if self.labels is not None:
    #         return self.labels

    #     f_path = self.data_dir + os.sep + "labels.txt"
    #     if not os.path.exists(f_path):
    #         logger.error("Labels file (labels.txt) not found in {}".format(self.data_dir))
    #         raise FileNotFoundError

    #     self.labels = []
    #     with open(f_path, encoding="utf-8") as fp:
    #         self.labels = [l.strip() for l in fp.readlines()]

    #     return self.labels

    @staticmethod
    def get_labels_filename():
        return "labels.txt"

    def get_vocabulary(self):
        examples = self.get_train_examples() + self.get_dev_examples() + self.get_test_examples()
        vocab = Vocabulary(start=1)
        for e in examples:
            for t in e.tokens:
                vocab.add(t)
        return vocab


class QAInputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    limit = 50
    i=0
    for entry in input_data:
        i+=1
        if i == limit:
            return examples
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = QuestionAnsweringInputExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens