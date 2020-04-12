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
import io
import logging
import os
import pickle
from typing import List
import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

from nlp_architect.data.sequential_tagging import TokenClsInputExample
from nlp_architect.models import TrainableModel
from nlp_architect.nn.torch.layers import CRF
from nlp_architect.nn.torch.distillation import TeacherStudentDistill
from nlp_architect.nn.torch.modules.embedders import IDCNN
from nlp_architect.utils.metrics import tagging
from nlp_architect.utils.text import Vocabulary, char_to_id

logger = logging.getLogger(__name__)


class NeuralTagger(TrainableModel):
    """
    Simple neural tagging model
    Supports pytorch embedder models, multi-gpu training, KD from teacher models

    Args:
        embedder_model: pytorch embedder model (valid nn.Module model)
        word_vocab (Vocabulary): word vocabulary
        labels (List, optional): list of labels. Defaults to None
        use_crf (bool, optional): use CRF a the classifier (instead of Softmax). Defaults to False.
        device (str, optional): device backend. Defatuls to 'cpu'.
        n_gpus (int, optional): number of gpus. Default to 0.
    """

    def __init__(
        self,
        embedder_model,
        word_vocab: Vocabulary,
        labels: List[str] = None,
        use_crf: bool = False,
        device: str = "cpu",
        n_gpus=0,
    ):
        super(NeuralTagger, self).__init__()
        self.model = embedder_model
        self.labels = labels
        self.num_labels = len(labels) + 1  # +1 for padding
        self.label_str_id = {l: i for i, l in enumerate(self.labels, 1)}
        self.label_id_str = {v: k for k, v in self.label_str_id.items()}
        self.word_vocab = word_vocab
        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=True)
        self.device = device
        self.n_gpus = n_gpus
        self.to(self.device, self.n_gpus)

    def convert_to_tensors(
        self,
        examples: List[TokenClsInputExample],
        max_seq_length: int = 128,
        max_word_length: int = 12,
        pad_id: int = 0,
        labels_pad_id: int = 0,
        include_labels: bool = True,
    ) -> TensorDataset:
        """
        Convert examples to valid tagger dataset

        Args:
            examples (List[TokenClsInputExample]): List of examples
            max_seq_length (int, optional): max words per sentence. Defaults to 128.
            max_word_length (int, optional): max characters in a word. Defaults to 12.
            pad_id (int, optional): padding int id. Defaults to 0.
            labels_pad_id (int, optional): labels padding id. Defaults to 0.
            include_labels (bool, optional): include labels in dataset. Defaults to True.

        Returns:
            TensorDataset: TensorDataset for given examples
        """

        features = []
        for example in examples:
            word_tokens = [self.word_vocab[t] for t in example.tokens]
            labels = []
            if include_labels:
                labels = [self.label_str_id.get(l) for l in example.label]
            word_chars = []
            for word in example.tokens:
                word_chars.append([char_to_id(c) for c in word])
            word_shapes = example.shapes

            # cut up to max length
            word_tokens = word_tokens[:max_seq_length]
            word_shapes = word_shapes[:max_seq_length]
            if include_labels:
                labels = labels[:max_seq_length]
            word_chars = word_chars[:max_seq_length]
            for i in range(len(word_chars)):
                word_chars[i] = word_chars[i][:max_word_length]
            mask = [1] * len(word_tokens)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(word_tokens)
            input_ids = word_tokens + ([pad_id] * padding_length)
            shape_ids = word_shapes + ([pad_id] * padding_length)
            mask = mask + ([0] * padding_length)
            if include_labels:
                label_ids = labels + ([labels_pad_id] * padding_length)

            word_char_ids = []
            # pad word vectors
            for i in range(len(word_chars)):
                word_char_ids.append(
                    word_chars[i] + ([pad_id] * (max_word_length - len(word_chars[i])))
                )

            # pad word vectors with remaining zero vectors
            for _ in range(padding_length):
                word_char_ids.append(([pad_id] * max_word_length))

            assert len(input_ids) == max_seq_length
            assert len(shape_ids) == max_seq_length
            if include_labels:
                assert len(label_ids) == max_seq_length
            assert len(word_char_ids) == max_seq_length
            for i in range(len(word_char_ids)):
                assert len(word_char_ids[i]) == max_word_length

            features.append(
                InputFeatures(
                    input_ids,
                    word_char_ids,
                    shape_ids,
                    mask=mask,
                    label_id=label_ids if include_labels else None,
                )
            )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_char_ids = torch.tensor([f.char_ids for f in features], dtype=torch.long)
        all_shape_ids = torch.tensor([f.shape_ids for f in features], dtype=torch.long)
        masks = torch.tensor([f.mask for f in features], dtype=torch.long)

        if include_labels:
            is_labeled = torch.tensor([True for f in features], dtype=torch.bool)
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_char_ids, all_shape_ids, masks, is_labeled, all_label_ids
            )
        else:
            is_labeled = torch.tensor([False for f in features], dtype=torch.bool)
            dataset = TensorDataset(all_input_ids, all_char_ids, all_shape_ids, masks, is_labeled)
        return dataset

    def get_optimizer(self, opt_fn=None, lr: int = 0.001):
        """
        Get default optimizer

        Args:
            lr (int, optional): learning rate. Defaults to 0.001.

        Returns:
            torch.optim.Optimizer: optimizer
        """
        params = self.model.parameters()
        if self.use_crf:
            params = list(params) + list(self.crf.parameters())
        if opt_fn is None:
            opt_fn = optim.Adam
        return opt_fn(params, lr=lr)

    @staticmethod
    def batch_mapper(batch):
        """
        Map batch to correct input names
        """
        mapping = {
            "words": batch[0],
            "word_chars": batch[1],
            "shapes": batch[2],
            "mask": batch[3],
            "is_labeled": batch[4],
        }
        if len(batch) == 6:
            mapping.update({"labels": batch[5]})
        return mapping

    def train(
        self,
        train_data_set: DataLoader,
        dev_data_set: DataLoader = None,
        test_data_set: DataLoader = None,
        epochs: int = 3,
        batch_size: int = 8,
        optimizer=None,
        max_grad_norm: float = 5.0,
        logging_steps: int = 50,
        save_steps: int = 100,
        save_path: str = None,
        distiller: TeacherStudentDistill = None,
        best_result_file: str = None,
        word_dropout: float = 0,
    ):
        """
        Train a tagging model

        Args:
            train_data_set (DataLoader): train examples dataloader.
                - If distiller object is provided train examples should contain a tuple of
                  student/teacher data examples.
            dev_data_set (DataLoader, optional): dev examples dataloader. Defaults to None.
            test_data_set (DataLoader, optional): test examples dataloader. Defaults to None.
            epochs (int, optional): num of epochs to train. Defaults to 3.
            batch_size (int, optional): batch size. Defaults to 8.
            optimizer (fn, optional): optimizer function. Defaults to default model optimizer.
            max_grad_norm (float, optional): max gradient norm. Defaults to 5.0.
            logging_steps (int, optional): number of steps between logging. Defaults to 50.
            save_steps (int, optional): number of steps between model saves. Defaults to 100.
            save_path (str, optional): model output path. Defaults to None.
            distiller (TeacherStudentDistill, optional): KD model for training the model using
            a teacher model. Defaults to None.
            best_result_file (str, optional): path to save best dev results when it's updated.
            word_dropout (float, optional): whole-word (-> oov) dropout rate. Defaults to 0.
        """
        if optimizer is None:
            optimizer = self.get_optimizer()
        train_batch_size = batch_size * max(1, self.n_gpus)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data_set.dataset))
        logger.info("  Num Epochs = %d", epochs)
        logger.info("  Instantaneous batch size per GPU/CPU = %d", batch_size)
        logger.info("  Total batch size = %d", train_batch_size)
        global_step = 0
        best_dev = 0
        dev_test = 0
        self.model.zero_grad()
        epoch_it = trange(epochs, desc="Epoch")
        for epoch in epoch_it:
            step_it = tqdm(train_data_set, desc="Train iteration")
            avg_loss = 0

            for step, batches in enumerate(step_it):
                self.model.train()

                batch, t_batch = (batches, []) if not distiller else (batches[:2])
                batch = tuple(t.to(self.device) for t in batch)
                inputs = self.batch_mapper(batch)
                logits = self.model(**inputs)

                if distiller:
                    t_batch = tuple(t.to(self.device) for t in t_batch)
                    t_logits = distiller.get_teacher_logits(t_batch)
                    valid_positions = (
                        t_batch[3] != 0.0
                    )  # TODO: implement method to get only valid logits from the model itself
                    valid_t_logits = {}
                    max_seq_len = logits.shape[1]
                    for i in range(len(logits)):  # each example in batch
                        valid_logit_i = t_logits[i][valid_positions[i]]
                        valid_t_logits[i] = (
                            valid_logit_i
                            if valid_logit_i.shape[0] <= max_seq_len
                            else valid_logit_i[:][:max_seq_len]
                        )  # cut to max len

                    # prepare teacher labels for non-labeled examples
                    t_labels_dict = {}
                    for i in range(len(valid_t_logits.keys())):
                        t_labels_dict[i] = torch.argmax(
                            F.log_softmax(valid_t_logits[i], dim=-1), dim=-1
                        )

                # pseudo labeling
                for i, is_labeled in enumerate(inputs["is_labeled"]):
                    if not is_labeled:
                        t_labels_i = t_labels_dict[i]
                        # add the padded teacher label:
                        inputs["labels"][i] = torch.cat(
                            (
                                t_labels_i,
                                torch.zeros([max_seq_len - len(t_labels_i)], dtype=torch.long).to(
                                    self.device
                                ),
                            ),
                            0,
                        )

                # apply word dropout to the input
                if word_dropout != 0:
                    tokens = inputs["words"]
                    tokens = np.array(tokens.detach().cpu())
                    word_probs = np.random.random(tokens.shape)
                    drop_indices = np.where(
                        (word_probs > word_dropout) & (tokens != 0)
                    )  # ignore padding indices
                    inputs["words"][drop_indices[0], drop_indices[1]] = self.word_vocab.oov_id

                # loss
                if self.use_crf:
                    loss = -1.0 * self.crf(logits, inputs["labels"], mask=inputs["mask"] != 0.0)
                else:
                    loss_fn = CrossEntropyLoss(ignore_index=0)
                    loss = loss_fn(logits.view(-1, self.num_labels), inputs["labels"].view(-1))

                # for idcnn training - add dropout penalty loss
                module = self.model.module if self.n_gpus > 1 else self.model
                if isinstance(module, IDCNN) and module.drop_penalty != 0:
                    logits_no_drop = self.model(**inputs, no_dropout=True)
                    sub = logits.sub(logits_no_drop)
                    drop_loss = torch.div(torch.sum(torch.pow(sub, 2)), 2)
                    loss += module.drop_penalty * drop_loss

                if self.n_gpus > 1:
                    loss = loss.mean()

                # add distillation loss if activated
                if distiller:
                    # filter masked student logits (no padding)
                    valid_s_logits = {}
                    valid_s_positions = inputs["mask"] != 0.0
                    for i in range(len(logits)):
                        valid_s_logit_i = logits[i][valid_s_positions[i]]
                        valid_s_logits[i] = valid_s_logit_i
                    loss = distiller.distill_loss_dict(loss, valid_s_logits, valid_t_logits)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                avg_loss += loss.item()
                if global_step % logging_steps == 0:
                    if step != 0:
                        logger.info(
                            " global_step = %s, average loss = %s", global_step, avg_loss / step
                        )
                        best_dev, dev_test = self.update_best_model(
                            dev_data_set,
                            test_data_set,
                            best_dev,
                            dev_test,
                            best_result_file,
                            avg_loss / step,
                            epoch,
                            save_path=None,
                        )
                if save_steps != 0 and save_path is not None and global_step % save_steps == 0:
                    self.save_model(save_path)
        self.update_best_model(
            dev_data_set,
            test_data_set,
            best_dev,
            dev_test,
            best_result_file,
            "end_training",
            "end_training",
            save_path=save_path + "/best_dev",
        )

    def _get_eval(self, ds, set_name):
        if ds is not None:
            logits, out_label_ids = self.evaluate(ds)
            res = self.evaluate_predictions(logits, out_label_ids)
            logger.info(" {} set F1 = {}".format(set_name, res["f1"]))
            return res["f1"]
        return None

    def to(self, device="cpu", n_gpus=0):
        """
        Put model on given device

        Args:
            device (str, optional): device backend. Defaults to 'cpu'.
            n_gpus (int, optional): number of gpus. Defaults to 0.
        """
        if self.model is not None:
            self.model.to(device)
            if self.use_crf:
                self.crf.to(device)
            if n_gpus > 1:
                self.model = torch.nn.DataParallel(self.model)
                if self.use_crf:
                    self.crf = torch.nn.DataParallel(self.crf)
        self.device = device
        self.n_gpus = n_gpus

    def evaluate(self, data_set: DataLoader):
        """
        Run evaluation on given dataloader

        Args:
            data_set (DataLoader): a data loader to run evaluation on

        Returns:
            logits, labels (if labels are given)
        """
        logger.info("***** Running inference *****")
        logger.info(" Batch size: {}".format(data_set.batch_size))
        preds = None
        out_label_ids = None
        for batch in tqdm(data_set, desc="Inference iteration"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = self.batch_mapper(batch)
                logits = self.model(**inputs)
            model_output = logits.detach().cpu()
            model_out_label_ids = inputs["labels"].detach().cpu() if "labels" in inputs else None
            if preds is None:
                preds = model_output
                out_label_ids = model_out_label_ids
            else:
                preds = torch.cat((preds, model_output), dim=0)
                out_label_ids = (
                    torch.cat((out_label_ids, model_out_label_ids), dim=0)
                    if out_label_ids is not None
                    else None
                )
        output = (preds,)
        if out_label_ids is not None:
            output = output + (out_label_ids,)
        return output

    def evaluate_predictions(self, logits, label_ids):
        """
        Evaluate given logits on truth labels

        Args:
            logits: logits of model
            label_ids: truth label ids

        Returns:
            dict: dictionary containing P/R/F1 metrics
        """
        active_positions = label_ids.view(-1) != 0.0
        active_labels = label_ids.view(-1)[active_positions]
        if self.use_crf:
            logits_shape = logits.size()
            decode_ap = active_positions.view(logits_shape[0], logits_shape[1]) != 0.0
            if self.n_gpus > 1:
                decode_fn = self.crf.module.decode
            else:
                decode_fn = self.crf.decode
            logits = decode_fn(logits.to(self.device), mask=decode_ap.to(self.device))
            logits = [l for ll in logits for l in ll]
        else:
            active_logits = logits.view(-1, len(self.label_id_str) + 1)[active_positions]
            logits = torch.argmax(F.log_softmax(active_logits, dim=1), dim=1)
            logits = logits.detach().cpu().numpy()
        out_label_ids = active_labels.detach().cpu().numpy()
        y_true, y_pred = self.extract_labels(out_label_ids, logits)
        p, r, f1 = tagging(y_pred, y_true)
        return {"p": p, "r": r, "f1": f1}

    def update_best_model(
        self,
        dev_data_set,
        test_data_set,
        best_dev,
        best_dev_test,
        best_result_file,
        loss,
        epoch,
        save_path=None,
    ):
        new_best_dev = best_dev
        new_test = best_dev_test
        dev = self._get_eval(dev_data_set, "dev")
        test = self._get_eval(test_data_set, "test")
        if dev > best_dev:
            new_best_dev = dev
            new_test = test
            if best_result_file is not None:
                with open(best_result_file, "a+") as f:
                    f.write(
                        "best dev= "
                        + str(new_best_dev)
                        + ", test= "
                        + str(new_test)
                        + ", loss= "
                        + str(loss)
                        + ", epoch= "
                        + str(epoch)
                        + "\n"
                    )
        logger.info("Best result: Dev=%s, Test=%s", str(new_best_dev), str(new_test))
        if save_path is not None:
            self.save_model(save_path)
        return new_best_dev, new_test

    def extract_labels(self, label_ids, logits):
        label_map = self.label_id_str
        y_true = []
        y_pred = []
        for p, y in zip(logits, label_ids):
            y_pred.append(label_map.get(p, "O"))
            y_true.append(label_map.get(y, "O"))
        assert len(y_true) == len(y_pred)
        return (y_true, y_pred)

    def inference(self, examples: List[TokenClsInputExample], batch_size: int = 64):
        """
        Do inference on given examples

        Args:
            examples (List[TokenClsInputExample]): examples
            batch_size (int, optional): batch size. Defaults to 64.

        Returns:
            List(tuple): a list of tuples of tokens, tags predicted by model
        """
        data_set = self.convert_to_tensors(examples, include_labels=False)
        inf_sampler = SequentialSampler(data_set)
        inf_dataloader = DataLoader(data_set, sampler=inf_sampler, batch_size=batch_size)
        logits = self.evaluate(inf_dataloader)
        active_positions = data_set.tensors[-1].view(len(data_set), -1) != 0.0
        logits = torch.argmax(F.log_softmax(logits[0], dim=2), dim=2)
        res_ids = []
        for i in range(logits.size()[0]):
            res_ids.append(logits[i][active_positions[i]].detach().cpu().numpy())
        output = []
        for tag_ids, ex in zip(res_ids, examples):
            tokens = ex.tokens
            tags = [self.label_id_str.get(t, "O") for t in tag_ids]
            output.append((tokens, tags))
        return output

    def save_model(self, output_dir: str):
        """
        Save model to path

        Args:
            output_dir (str): output directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.model, os.path.join(output_dir, "model.bin"))
        if self.use_crf:
            torch.save(self.crf, os.path.join(output_dir, "crf.bin"))
        with io.open(output_dir + os.sep + "labels.txt", "w", encoding="utf-8") as fw:
            for l in self.labels:
                fw.write("{}\n".format(l))
        with io.open(output_dir + os.sep + "w_vocab.dat", "wb") as fw:
            pickle.dump(self.word_vocab, fw)

    @classmethod
    def load_model(cls, model_path: str):
        """
        Load a tagger model from given path

        Args:
            model_path (str): model path

            NeuralTagger: tagger model loaded from path
        """
        # Load a trained model and vocabulary from given path
        if not os.path.exists(model_path):
            raise FileNotFoundError
        with io.open(model_path + os.sep + "labels.txt") as fp:
            labels = [l.strip() for l in fp.readlines()]

        with io.open(model_path + os.sep + "w_vocab.dat", "rb") as fp:
            w_vocab = pickle.load(fp)
        # load model.bin into
        model_file_path = model_path + os.sep + "model.bin"
        if not os.path.exists(model_file_path):
            raise FileNotFoundError
        model = torch.load(model_file_path)
        new_class = cls(model, w_vocab, labels)
        crf_file_path = model_path + os.sep + "crf.bin"
        if os.path.exists(crf_file_path):
            new_class.use_crf = True
            new_class.crf = torch.load(crf_file_path)
        else:
            new_class.use_crf = False
        return new_class

    def get_logits(self, batch):
        self.model.eval()
        inputs = self.batch_mapper(batch)
        outputs = self.model(**inputs)
        return outputs[-1]


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, char_ids, shape_ids, mask=None, label_id=None):
        self.input_ids = input_ids
        self.char_ids = char_ids
        self.shape_ids = shape_ids
        self.mask = mask
        self.label_id = label_id
