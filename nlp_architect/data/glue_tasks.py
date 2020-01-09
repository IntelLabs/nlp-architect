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
import logging
import os

from nlp_architect.data.sequence_classification import SequenceClsInputExample
from nlp_architect.data.utils import DataProcessor, Task, read_tsv

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            if set_type in ["train", "dev"]:
                label = line[0]
                examples.append(
                    SequenceClsInputExample(guid=guid, text=text_a, text_b=text_b, label=label)
                )
            else:
                examples.append(SequenceClsInputExample(guid=guid, text=text_a, text_b=text_b))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched"
        )

    def get_test_examples(self, data_dir):
        return self._create_examples(
            read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched"
        )

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            if set_type in ["train", "dev_matched"]:
                label = line[-1]
                examples.append(
                    SequenceClsInputExample(guid=guid, text=text_a, text_b=text_b, label=label)
                )
            else:
                examples.append(SequenceClsInputExample(guid=guid, text=text_a, text_b=text_b))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_matched"
        )

    def get_test_examples(self, data_dir):
        return self._create_examples(
            read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched"
        )


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 and set_type not in ["train", "dev"]:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type in ["train", "dev"]:
                text_a = line[3]
                label = line[1]
                examples.append(
                    SequenceClsInputExample(guid=guid, text=text_a, text_b=None, label=label)
                )
            else:
                text_a = line[1]
                examples.append(SequenceClsInputExample(guid=guid, text=text_a))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type in ["train", "dev"]:
                text_a = line[0]
                label = line[1]
                examples.append(
                    SequenceClsInputExample(guid=guid, text=text_a, text_b=None, label=label)
                )
            else:
                text_a = line[1]
                examples.append(SequenceClsInputExample(guid=guid, text=text_a))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return [None]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            if set_type in ["train", "dev"]:
                label = line[-1]
                examples.append(
                    SequenceClsInputExample(guid=guid, text=text_a, text_b=text_b, label=label)
                )
            else:
                examples.append(SequenceClsInputExample(guid=guid, text=text_a, text_b=text_b))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type in ["train", "dev"]:
                try:
                    text_a = line[3]
                    text_b = line[4]
                    label = line[5]
                except IndexError:
                    continue
                examples.append(
                    SequenceClsInputExample(guid=guid, text=text_a, text_b=text_b, label=label)
                )
            else:
                try:
                    text_a = line[1]
                    text_b = line[2]
                except IndexError:
                    continue
                examples.append(SequenceClsInputExample(guid=guid, text=text_a, text_b=text_b))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["entailment", "not_entailment"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            if set_type in ["train", "dev"]:
                label = line[-1]
                examples.append(
                    SequenceClsInputExample(guid=guid, text=text_a, text_b=text_b, label=label)
                )
            else:
                examples.append(SequenceClsInputExample(guid=guid, text=text_a, text_b=text_b))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["entailment", "not_entailment"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            if set_type in ["train", "dev"]:
                label = line[-1]
                examples.append(
                    SequenceClsInputExample(guid=guid, text=text_a, text_b=text_b, label=label)
                )
            else:
                examples.append(SequenceClsInputExample(guid=guid, text=text_a, text_b=text_b))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            if set_type in ["train", "dev"]:
                label = line[-1]
                examples.append(
                    SequenceClsInputExample(guid=guid, text=text_a, text_b=text_b, label=label)
                )
            else:
                examples.append(SequenceClsInputExample(guid=guid, text=text_a, text_b=text_b))
        return examples


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    output_mode,
    cls_token_at_end=False,
    pad_on_left=False,
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_token=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    cls_token_segment_id=1,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token
        (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example {} of {}".format(ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[: (max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}

DEFAULT_FOLDER_NAMES = {
    "cola": "CoLA",
    "sst": "SST-2",
    "mrpc": "MRPC",
    "stsb": "STS-B",
    "qqp": "QQP",
    "mnli": "MNLI",
    "qnli": "QNLI",
    "rte": "RTE",
    "wnli": "WNLI",
    "snli": "SNLI",
}


def get_glue_task(task_name: str, data_dir: str = None):
    """Return a GLUE task object
    Args:
        task_name (str): name of GLUE task
        data_dir (str, optional): path to dataset, if not provided will be taken from
            GLUE_DIR env. variable
    """
    task_name = task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: {}".format(task_name))
    task_processor = processors[task_name]()
    if data_dir is None:
        data_dir = os.path.join(os.environ["GLUE_DIR"], DEFAULT_FOLDER_NAMES[task_name])
    task_type = output_modes[task_name]
    return Task(task_name, task_processor, data_dir, task_type)
