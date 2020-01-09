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

            # cut up to max length
            word_tokens = word_tokens[:max_seq_length]
            if include_labels:
                labels = labels[:max_seq_length]
            word_chars = word_chars[:max_seq_length]
            for i in range(len(word_chars)):
                word_chars[i] = word_chars[i][:max_word_length]
            mask = [1] * len(word_tokens)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(word_tokens)
            input_ids = word_tokens + ([pad_id] * padding_length)
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
            if include_labels:
                assert len(label_ids) == max_seq_length
            assert len(word_char_ids) == max_seq_length
            for i in range(len(word_char_ids)):
                assert len(word_char_ids[i]) == max_word_length

            features.append(
                InputFeatures(
                    input_ids,
                    word_char_ids,
                    mask=mask,
                    label_id=label_ids if include_labels else None,
                )
            )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_char_ids = torch.tensor([f.char_ids for f in features], dtype=torch.long)
        masks = torch.tensor([f.mask for f in features], dtype=torch.long)

        if include_labels:
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_char_ids, masks, all_label_ids)
        else:
            dataset = TensorDataset(all_input_ids, all_char_ids, masks)
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
        mapping = {"words": batch[0], "word_chars": batch[1], "mask": batch[2]}
        if len(batch) == 4:
            mapping.update({"labels": batch[3]})
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
    ):
        """
        Train a tagging model

        Args:
            train_data_set (DataLoader): train examples dataloader. If distiller object is
            provided train examples should contain a tuple of student/teacher data examples.
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
        self.model.zero_grad()
        epoch_it = trange(epochs, desc="Epoch")
        for _ in epoch_it:
            step_it = tqdm(train_data_set, desc="Train iteration")
            avg_loss = 0
            for step, batch in enumerate(step_it):
                self.model.train()
                if distiller:
                    batch, t_batch = batch[:2]
                    t_batch = tuple(t.to(self.device) for t in t_batch)
                    t_logits = distiller.get_teacher_logits(t_batch)
                batch = tuple(t.to(self.device) for t in batch)
                inputs = self.batch_mapper(batch)
                logits = self.model(**inputs)
                if self.use_crf:
                    loss = -1.0 * self.crf(logits, inputs["labels"], mask=inputs["mask"] != 0.0)
                else:
                    loss_fn = CrossEntropyLoss(ignore_index=0)
                    loss = loss_fn(logits.view(-1, self.num_labels), inputs["labels"].view(-1))
                if self.n_gpus > 1:
                    loss = loss.mean()

                # add distillation loss if activated
                if distiller:
                    loss = distiller.distill_loss(loss, logits, t_logits)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                optimizer.step()
                # self.model.zero_grad()
                optimizer.zero_grad()
                global_step += 1
                avg_loss += loss.item()
                if global_step % logging_steps == 0:
                    if step != 0:
                        logger.info(
                            " global_step = %s, average loss = %s", global_step, avg_loss / step
                        )
                    self._get_eval(dev_data_set, "dev")
                    self._get_eval(test_data_set, "test")
                if save_path is not None and global_step % save_steps == 0:
                    self.save_model(save_path)

    def train_pseudo(
        self,
        labeled_data_set: DataLoader,
        unlabeled_data_set: DataLoader,
        distiller: TeacherStudentDistill,
        dev_data_set: DataLoader = None,
        test_data_set: DataLoader = None,
        batch_size_l: int = 8,
        batch_size_ul: int = 8,
        epochs: int = 100,
        optimizer=None,
        max_grad_norm: float = 5.0,
        logging_steps: int = 50,
        save_steps: int = 100,
        save_path: str = None,
        save_best: bool = False,
    ):
        """
        Train a tagging model

        Args:
            train_data_set (DataLoader): train examples dataloader. If distiller object is
            provided train examples should contain a tuple of student/teacher data examples.
            dev_data_set (DataLoader, optional): dev examples dataloader. Defaults to None.
            test_data_set (DataLoader, optional): test examples dataloader. Defaults to None.
            batch_size_l (int, optional): batch size for the labeled dataset. Defaults to 8.
            batch_size_ul (int, optional): batch size for the unlabeled dataset. Defaults to 8.
            epochs (int, optional): num of epochs to train. Defaults to 100.
            optimizer (fn, optional): optimizer function. Defaults to default model optimizer.
            max_grad_norm (float, optional): max gradient norm. Defaults to 5.0.
            logging_steps (int, optional): number of steps between logging. Defaults to 50.
            save_steps (int, optional): number of steps between model saves. Defaults to 100.
            save_path (str, optional): model output path. Defaults to None.
            save_best (str, optional): wether to save model when result is best on dev set
            distiller (TeacherStudentDistill, optional): KD model for training the model using
            a teacher model. Defaults to None.
        """
        if optimizer is None:
            optimizer = self.get_optimizer()
        train_batch_size_l = batch_size_l * max(1, self.n_gpus)
        train_batch_size_ul = batch_size_ul * max(1, self.n_gpus)
        logger.info("***** Running training *****")
        logger.info("  Num labeled examples = %d", len(labeled_data_set.dataset))
        logger.info("  Num unlabeled examples = %d", len(unlabeled_data_set.dataset))
        logger.info("  Instantaneous labeled batch size per GPU/CPU = %d", batch_size_l)
        logger.info("  Instantaneous unlabeled batch size per GPU/CPU = %d", batch_size_ul)
        logger.info("  Total batch size labeled= %d", train_batch_size_l)
        logger.info("  Total batch size unlabeled= %d", train_batch_size_ul)
        global_step = 0
        self.model.zero_grad()
        avg_loss = 0
        iter_l = iter(labeled_data_set)
        iter_ul = iter(unlabeled_data_set)
        epoch_l = 0
        epoch_ul = 0
        s_idx = -1
        best_dev = 0
        best_test = 0
        while True:
            logger.info("labeled epoch=%d, unlabeled epoch=%d", epoch_l, epoch_ul)
            loss_labeled = 0
            loss_unlabeled = 0
            try:
                batch_l = next(iter_l)
                s_idx += 1
            except StopIteration:
                iter_l = iter(labeled_data_set)
                epoch_l += 1
                batch_l = next(iter_l)
                s_idx = 0
                avg_loss = 0
            try:
                batch_ul = next(iter_ul)
            except StopIteration:
                iter_ul = iter(unlabeled_data_set)
                epoch_ul += 1
                batch_ul = next(iter_ul)
            if epoch_ul > epochs:
                logger.info("Done")
                return
            self.model.train()
            batch_l, t_batch_l = batch_l[:2]
            batch_ul, t_batch_ul = batch_ul[:2]
            t_batch_l = tuple(t.to(self.device) for t in t_batch_l)
            t_batch_ul = tuple(t.to(self.device) for t in t_batch_ul)
            t_logits = distiller.get_teacher_logits(t_batch_l)
            t_logits_ul = distiller.get_teacher_logits(t_batch_ul)
            batch_l = tuple(t.to(self.device) for t in batch_l)
            batch_ul = tuple(t.to(self.device) for t in batch_ul)
            inputs = self.batch_mapper(batch_l)
            inputs_ul = self.batch_mapper(batch_ul)
            logits = self.model(**inputs)
            logits_ul = self.model(**inputs_ul)
            t_labels = torch.argmax(F.log_softmax(t_logits_ul, dim=2), dim=2)
            if self.use_crf:
                loss_labeled = -1.0 * self.crf(logits, inputs["labels"], mask=inputs["mask"] != 0.0)
                loss_unlabeled = -1.0 * self.crf(logits_ul, t_labels, mask=inputs_ul["mask"] != 0.0)
            else:
                loss_fn = CrossEntropyLoss(ignore_index=0)
                loss_labeled = loss_fn(logits.view(-1, self.num_labels), inputs["labels"].view(-1))
                loss_unlabeled = loss_fn(logits_ul.view(-1, self.num_labels), t_labels.view(-1))

            if self.n_gpus > 1:
                loss_labeled = loss_labeled.mean()
                loss_unlabeled = loss_unlabeled.mean()

            # add distillation loss
            loss_labeled = distiller.distill_loss(loss_labeled, logits, t_logits)
            loss_unlabeled = distiller.distill_loss(loss_unlabeled, logits_ul, t_logits_ul)

            # sum labeled and unlabeled losses
            loss = loss_labeled + loss_unlabeled
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            optimizer.step()
            # self.model.zero_grad()
            optimizer.zero_grad()
            global_step += 1
            avg_loss += loss.item()
            if global_step % logging_steps == 0:
                if s_idx != 0:
                    logger.info(
                        " global_step = %s, average loss = %s", global_step, avg_loss / s_idx
                    )
                dev = self._get_eval(dev_data_set, "dev")
                test = self._get_eval(test_data_set, "test")
                if dev > best_dev:
                    best_dev = dev
                    best_test = test
                    if save_path is not None and save_best:
                        self.save_model(save_path)
                logger.info("Best result: dev= %s, test= %s", str(best_dev), str(best_test))
            if save_path is not None and global_step % save_steps == 0:
                self.save_model(save_path)

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
        eval_loss = 0.0
        preds = None
        out_label_ids = None
        for batch in tqdm(data_set, desc="Inference iteration"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = self.batch_mapper(batch)
                logits = self.model(**inputs)
                if "labels" in inputs:
                    if self.use_crf:
                        loss = -1.0 * self.crf(logits, inputs["labels"], mask=inputs["mask"] != 0.0)
                    else:
                        loss_fn = CrossEntropyLoss(ignore_index=0)
                        loss = loss_fn(logits.view(-1, self.num_labels), inputs["labels"].view(-1))
                    eval_loss += loss.mean().item()
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

    def __init__(self, input_ids, char_ids, mask=None, label_id=None):
        self.input_ids = input_ids
        self.char_ids = char_ids
        self.mask = mask
        self.label_id = label_id
