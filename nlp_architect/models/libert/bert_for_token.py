# ******************************************************************************
# Copyright 2019-2020 Intel Corporation
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
"""BERT-based model for token classification."""

# pylint: disable=no-member, not-callable, attribute-defined-outside-init, arguments-differ, missing-function-docstring
# pylint: disable=too-many-ancestors, too-many-instance-attributes, too-many-arguments
import os
from argparse import Namespace
from pathlib import Path
from collections import OrderedDict
from os.path import realpath
from torch.nn import CrossEntropyLoss
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import _logger as log
import torch
from torch.utils.data import DataLoader, TensorDataset
torch.multiprocessing.set_sharing_strategy('file_system')
from seqeval.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             performance_measure)
from transformers import (
    BertForTokenClassification,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    AdamW
)

import absa_utils
from libert_model import LiBertForToken, LiBertConfig
#from libert_model_14heads import LiBertForToken, LiBertConfig

LIBERT_DIR = Path(realpath(__file__)).parent

MODEL_CONFIG = {
    'bert': (BertForTokenClassification, BertConfig, BertTokenizer),
    'libert': (LiBertForToken, LiBertConfig, BertTokenizer)
}

# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

class BertForToken(pl.LightningModule):
    """Lightning module for BERT for token classification."""
    def __init__(self, hparams):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        data_root = Path(__file__).parent.absolute() / 'data' / 'csv'

        self.labels = absa_utils.get_labels(data_root / hparams.labels)
        num_labels = len(self.labels)
        hparams.data_dir = data_root / hparams.data_dir

        if not hparams.cache_dir:
            hparams.cache_dir = LIBERT_DIR / 'cache'
        if not hparams.output_dir:
            hparams.output_dir = LIBERT_DIR / 'models'

        self.model_type, self.config_type, self.tokenizer_type = MODEL_CONFIG[hparams.model_type]

        self.config = self.config_type \
            .from_pretrained(hparams.model_name_or_path,
                             **({"num_labels": num_labels} if num_labels is not None else {}))

        if hasattr(self.config, 'add_extra_args') and callable(self.config.add_extra_args):
            self.config.add_extra_args(hparams)

        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.tokenizer = self.tokenizer_type \
            .from_pretrained(hparams.model_name_or_path, cache_dir=hparams.cache_dir)

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.step_count = 0
        # self.tfmr_ckpts = {}

        self.model = self.model_type.from_pretrained(
            hparams.model_name_or_path,
            from_tf=bool(".ckpt" in hparams.model_name_or_path),
            config=self.config,
            cache_dir=hparams.cache_dir)

        self.hparams = hparams
        self.sentence_metrics = None

    def forward(self, **inputs):
        return self.model(**inputs)

    def prepare_data(self):
        "Called to initialize data. Use the call to construct features"
        for mode in "train", "dev", "test":
            cached_features_file = self._feature_file(mode)
            if os.path.exists(cached_features_file) and not self.hparams.overwrite_cache:
                log.debug("Loading features from cached file %s", cached_features_file)
                features = torch.load(cached_features_file)
            else:
                log.debug("Creating features from dataset file at %s", self.hparams.data_dir)
                examples = absa_utils.read_examples_from_file(self.hparams.data_dir, mode)
                features = absa_utils.convert_examples_to_features(
                    examples,
                    self.labels,
                    self.hparams.max_seq_length,
                    self.tokenizer
                )
                log.debug("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

    @staticmethod
    def map_to_inputs(batch):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                  "labels": batch[3]}
        if len(batch) >= 5:
            inputs["parse"] = batch[4]
        if len(batch) >= 6:
            inputs["heads_idx"] = batch[5]
        return inputs

    def training_step(self, batch, _):
        "Compute loss and log."
        inputs = self.map_to_inputs(batch)
        outputs, detailed_loss = self(**inputs)
        cls_loss, aux_loss = detailed_loss
        loss = outputs[0]
        tensorboard_logs = {'train_loss_step': loss, "train_cls_loss_step": cls_loss.detach().cpu(),
                            "train_aux_loss_step": aux_loss.detach().cpu()}
        return {'loss': loss, 'log': tensorboard_logs, 'cls_loss': cls_loss, 'aux_loss': aux_loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_cls_loss = torch.stack([x['cls_loss'] for x in outputs]).mean()
        avg_aux_loss = torch.stack([x['aux_loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss, 'step': self.current_epoch, 'train_cls_loss': avg_cls_loss, 'train_aux_loss': avg_aux_loss}
        return {'loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, _):
        "Compute validation."
        inputs = self.map_to_inputs(batch)
        outputs, detailed_loss = self(**inputs)
        cls_loss, aux_loss = detailed_loss
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu()
        target = inputs["labels"].detach().cpu()
        return {"val_loss_step": tmp_eval_loss.detach().cpu(), "pred": preds, "target": target,
                "val_cls_loss_step": cls_loss.detach().cpu(), "val_aux_loss_step": aux_loss.detach().cpu()}

    def validation_epoch_end(self, outputs):
        ret, _, _ = self._eval_end(outputs)
        logs = ret["log"]
        logs['step'] = self.current_epoch
        return {"val_loss": logs["val_loss"], "log": logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        ret, _, _ = self._eval_end(outputs)
        logs = ret["log"]
        logs['step'] = self.current_epoch
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"test_loss": logs["val_loss"], "log": logs}

    def _eval_end(self, outputs):
        "Evaluation called for both Val and Test"
        val_loss_mean = torch.stack([x["val_loss_step"] for x in outputs]).mean()
        val_cls_loss_mean = torch.stack([x["val_cls_loss_step"] for x in outputs]).mean()
        val_aux_loss_mean = torch.stack([x["val_aux_loss_step"] for x in outputs]).mean()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)
        preds = np.argmax(preds, axis=2)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        label_map = dict(enumerate(self.labels))
        target = [[] for _ in range(out_label_ids.shape[0])]
        pred = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    target[i].append(label_map[out_label_ids[i][j]])
                    pred[i].append(label_map[preds[i][j]])

        calc = lambda f: torch.tensor(f(target, pred))
        results = OrderedDict({
            "val_loss": val_loss_mean,
            "val_cls_loss": val_cls_loss_mean,
            "val_aux_loss": val_aux_loss_mean,
            "micro_precision": calc(precision_score),
            "micro_recall": calc(recall_score),
            "micro_f1": calc(f1_score),
            "micro_accuracy": calc(accuracy_score)
        })
        confusion = performance_measure(target, pred)
        type_metrics, macro_avg = absa_utils.detailed_metrics(target, pred)
        results.update(type_metrics)
        results.update(macro_avg)
        results.update(confusion)

        per_sentence = lambda f: [f([t], [p]) for t, p in zip(target, pred)]
        self.sentence_metrics = {
            "precision": per_sentence(precision_score),
            "recall": per_sentence(recall_score),
            "f1": per_sentence(f1_score),
            "accuracy": per_sentence(accuracy_score)
        }
        ret = results.copy()
        ret["log"] = results
        return ret, pred, target

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps()
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def setup(self, mode):
        if mode == "test":
            self.dataset_size = len(self.test_dataloader().dataset)
        else:
            self.train_loader = self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)
            self.dataset_size = len(self.train_dataloader().dataset)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def get_dataloader(self, mode, batch_size, shuffle):
        "Load datasets. Called after prepare data."
        cached_features_file = self._feature_file(mode)
        log.debug("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        if features[0].token_type_ids is not None:
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features],
                                              dtype=torch.long)
        else:
            all_token_type_ids = torch.tensor([0 for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        tensors = [all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids]

        if self.model_type is LiBertForToken:
            #### Attatch syntactic info ###
            dep_heads = torch.tensor([f.dep_heads for f in features], dtype=torch.float)
            tensors.append(dep_heads)

            head_idx = torch.tensor([f.head_idx for f in features], dtype=torch.long)
            tensors.append(head_idx)

        return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length)))

    def get_str(self) -> str:
        model_str = f'{self.hparams.model_type}'
        if self.hparams.model_type == 'libert' and self.hparams.rnd_init:
            model_str += '_rnd_init'
        return model_str

class LoggingCallback(pl.Callback):
    """Class for logging callbacks."""

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        print("***** Validation results *****")
        # Log results
        print(absa_utils.tabular(trainer.callback_metrics, 'Metrics'))

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        print("***** Test results *****")
        # Log results
        print(absa_utils.tabular(trainer.callback_metrics, 'Metrics'))

        log_dir = Path(trainer.logger.experiment.log_dir)
        with open(log_dir / 'sent_f1.txt', 'w', encoding='utf-8') as f1_file:
            f1_file.writelines([f'{v}\n' for v in pl_module.sentence_metrics['f1']])
