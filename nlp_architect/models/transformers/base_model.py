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
from typing import List, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    XLMConfig,
    XLMTokenizer,
    XLNetConfig,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from nlp_architect.models import TrainableModel
from nlp_architect.models.transformers.quantized_bert import QuantizedBertConfig

logger = logging.getLogger(__name__)


ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, XLNetConfig, XLMConfig)
    ),
    (),
)


def get_models(models: List[str]):
    if models is not None:
        return [m for m in ALL_MODELS if m.split("-")[0] in models]
    return ALL_MODELS


class TransformerBase(TrainableModel):
    """
    Transformers base model (for working with pytorch-transformers models)
    """

    MODEL_CONFIGURATIONS = {
        "bert": (BertConfig, BertTokenizer),
        "quant_bert": (QuantizedBertConfig, BertTokenizer),
        "xlnet": (XLNetConfig, XLNetTokenizer),
        "xlm": (XLMConfig, XLMTokenizer),
        "roberta": (RobertaConfig, RobertaTokenizer),
    }

    def __init__(
        self,
        model_type: str,
        model_name_or_path: str,
        labels: List[str] = None,
        num_labels: int = None,
        config_name=None,
        tokenizer_name=None,
        do_lower_case=False,
        output_path=None,
        device="cpu",
        n_gpus=0,
    ):
        """
        Transformers base model (for working with pytorch-transformers models)

        Args:
            model_type (str): transformer model type
            model_name_or_path (str): model name or path to model
            labels (List[str], optional): list of labels. Defaults to None.
            num_labels (int, optional): number of labels. Defaults to None.
            config_name ([type], optional): configuration name. Defaults to None.
            tokenizer_name ([type], optional): tokenizer name. Defaults to None.
            do_lower_case (bool, optional): lower case input words. Defaults to False.
            output_path ([type], optional): model output path. Defaults to None.
            device (str, optional): backend device. Defaults to 'cpu'.
            n_gpus (int, optional): num of gpus. Defaults to 0.

        Raises:
            FileNotFoundError: [description]
        """
        assert model_type in self.MODEL_CONFIGURATIONS.keys(), "unsupported model_type"
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.labels = labels
        self.num_labels = num_labels
        self.do_lower_case = do_lower_case
        if output_path is not None and not os.path.exists(output_path):
            raise FileNotFoundError("output_path is not found")
        self.output_path = output_path

        self.model_class = None
        config_class, tokenizer_class = self.MODEL_CONFIGURATIONS[model_type]
        self.config_class = config_class
        self.tokenizer_class = tokenizer_class

        self.tokenizer_name = tokenizer_name
        self.tokenizer = self._load_tokenizer(self.tokenizer_name)
        self.config_name = config_name
        self.config = self._load_config(config_name)

        self.model = None
        self.device = device
        self.n_gpus = n_gpus

        self._optimizer = None
        self._scheduler = None

    def to(self, device="cpu", n_gpus=0):
        if self.model is not None:
            self.model.to(device)
            if n_gpus > 1:
                self.model = torch.nn.DataParallel(self.model)
        self.device = device
        self.n_gpus = n_gpus

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, sch):
        self._scheduler = sch

    def setup_default_optimizer(
        self,
        weight_decay: float = 0.0,
        learning_rate: float = 5e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        total_steps: int = 0,
    ):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

    def _load_config(self, config_name=None):
        config = self.config_class.from_pretrained(
            config_name if config_name else self.model_name_or_path, num_labels=self.num_labels
        )
        return config

    def _load_tokenizer(self, tokenizer_name=None):
        tokenizer = self.tokenizer_class.from_pretrained(
            tokenizer_name if tokenizer_name else self.model_name_or_path,
            do_lower_case=self.do_lower_case,
        )
        return tokenizer

    def save_model(self, output_dir: str, save_checkpoint: bool = False, args=None):
        """
        Save model/tokenizer/arguments to given output directory

        Args:
            output_dir (str): path to output directory
            save_checkpoint (bool, optional): save as checkpoint. Defaults to False.
            args ([type], optional): arguments object to save. Defaults to None.
        """
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info("Saving model checkpoint to %s", output_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)
        if not save_checkpoint:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            with io.open(output_dir + os.sep + "labels.txt", "w", encoding="utf-8") as fw:
                for l in self.labels:
                    fw.write("{}\n".format(l))
            if args is not None:
                torch.save(args, os.path.join(output_dir, "training_args.bin"))

    @classmethod
    def load_model(cls, model_path: str, model_type: str, *args, **kwargs):
        """
        Create a TranformerBase deom from given path

        Args:
            model_path (str): path to model
            model_type (str): model type

        Returns:
            TransformerBase: model
        """
        # Load a trained model and vocabulary from given path
        if not os.path.exists(model_path):
            raise FileNotFoundError
        with io.open(model_path + os.sep + "labels.txt") as fp:
            labels = [l.strip() for l in fp.readlines()]
        return cls(
            model_type=model_type, model_name_or_path=model_path, labels=labels, *args, **kwargs
        )

    @staticmethod
    def get_train_steps_epochs(
        max_steps: int, num_train_epochs: int, gradient_accumulation_steps: int, num_samples: int
    ):
        """
        get train steps and epochs

        Args:
            max_steps (int): max steps
            num_train_epochs (int): num epochs
            gradient_accumulation_steps (int): gradient accumulation steps
            num_samples (int): number of samples

        Returns:
            Tuple: total steps, number of epochs
        """
        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (num_samples // gradient_accumulation_steps) + 1
        else:
            t_total = num_samples // gradient_accumulation_steps * num_train_epochs
        return t_total, num_train_epochs

    def get_logits(self, batch):
        self.model.eval()
        inputs = self._batch_mapper(batch)
        outputs = self.model(**inputs)
        return outputs[-1]

    def _train(
        self,
        data_set: DataLoader,
        dev_data_set: Union[DataLoader, List[DataLoader]] = None,
        test_data_set: Union[DataLoader, List[DataLoader]] = None,
        gradient_accumulation_steps: int = 1,
        per_gpu_train_batch_size: int = 8,
        max_steps: int = -1,
        num_train_epochs: int = 3,
        max_grad_norm: float = 1.0,
        logging_steps: int = 50,
        save_steps: int = 100,
    ):
        """Run model training
            batch_mapper: a function that maps a batch into parameters that the model
                          expects in the forward method (for use with custom heads and models).
                          If None it will default to the basic models input structure.
            logging_callback_fn: a function that is called in each evaluation step
                          with the model as a parameter.

        """
        t_total, num_train_epochs = self.get_train_steps_epochs(
            max_steps, num_train_epochs, gradient_accumulation_steps, len(data_set)
        )
        if self.optimizer is None and self.scheduler is None:
            logger.info("Loading default optimizer and scheduler")
            self.setup_default_optimizer(total_steps=t_total)

        train_batch_size = per_gpu_train_batch_size * max(1, self.n_gpus)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(data_set.dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per GPU/CPU = %d", per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            train_batch_size * gradient_accumulation_steps,
        )
        logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(num_train_epochs, desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(data_set, desc="Train iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = self._batch_mapper(batch)
                outputs = self.model(**inputs)
                loss = outputs[0]  # get loss

                if self.n_gpus > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        # Log metrics and run evaluation on dev/test
                        for ds in [dev_data_set, test_data_set]:
                            if ds is None:  # got no data loader
                                continue
                            if isinstance(ds, DataLoader):
                                ds = [ds]
                            for d in ds:
                                logits, label_ids = self._evaluate(d)
                                self.evaluate_predictions(logits, label_ids)
                        logger.info("lr = {}".format(self.scheduler.get_lr()[0]))
                        logger.info("loss = {}".format((tr_loss - logging_loss) / logging_steps))
                        logging_loss = tr_loss

                    if save_steps > 0 and global_step % save_steps == 0:
                        # Save model checkpoint
                        self.save_model_checkpoint(
                            output_path=self.output_path, name="checkpoint-{}".format(global_step)
                        )

                if 0 < max_steps < global_step:
                    epoch_iterator.close()
                    break
            if 0 < max_steps < global_step:
                train_iterator.close()
                break

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    def _evaluate(self, data_set: DataLoader):
        logger.info("***** Running inference *****")
        logger.info(" Batch size: {}".format(data_set.batch_size))
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(data_set, desc="Inference iteration"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = self._batch_mapper(batch)
                outputs = self.model(**inputs)
                if "labels" in inputs:
                    tmp_eval_loss, logits = outputs[:2]
                    eval_loss += tmp_eval_loss.mean().item()
                else:
                    logits = outputs[0]
            nb_eval_steps += 1
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
        if out_label_ids is None:
            return preds
        return preds, out_label_ids

    def _batch_mapper(self, batch):
        mapping = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            # XLM don't use segment_ids
            "token_type_ids": batch[2]
            if self.model_type in ["bert", "quant_bert", "xlnet"]
            else None,
        }
        if len(batch) == 4:
            mapping.update({"labels": batch[3]})
        return mapping

    def evaluate_predictions(self, logits, label_ids):
        raise NotImplementedError(
            "evaluate_predictions method must be implemented in order to"
            "be used for dev/test set evaluation"
        )

    def save_model_checkpoint(self, output_path: str, name: str):
        """
        save model checkpoint

        Args:
            output_path (str): output path
            name (str): name of checkpoint
        """
        output_dir_path = os.path.join(output_path, name)
        self.save_model(output_dir_path, save_checkpoint=True)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id=None, valid_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
