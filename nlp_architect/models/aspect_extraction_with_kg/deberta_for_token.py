# ******************************************************************************
# Copyright 2020-2021 Intel Corporation
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
"""DeBERTa-based model for token classification."""

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
from transformers import DebertaConfig, DebertaTokenizer
from deberta_model import DebertaForTokenClassification, CustomDebertaForTokenClassification, CustomDebertaConfig

LIBERT_DIR = Path(realpath(__file__)).parent

MODEL_CONFIG = {
    'bert': (BertForTokenClassification, BertConfig, BertTokenizer),
    'deberta': (DebertaForTokenClassification, DebertaConfig, DebertaTokenizer),
    'custom_deberta': (CustomDebertaForTokenClassification, CustomDebertaConfig, DebertaTokenizer),
}

log.setLevel("INFO")

class DebertaForToken(pl.LightningModule):
    """Lightning module for BERT for token classification."""
    def __init__(self, hparams):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        print("hparams:") 
        for hparam in vars(hparams):
            print(f"{hparam}: {getattr(hparams, hparam)}")
        print()

        data_root = Path(hparams.data_root)

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

        # Copying content projections as starting point for mark projections
        if getattr(hparams, "mark_projection_init", "random") == "content":
            for layer_no, layer in enumerate(self.model.base_model.encoder.layer):
                q_weight, k_weight, v_weight = layer.attention.self.in_proj.weight.detach().chunk(3, dim=0)
                layer.attention.self.mark_q_proj.state_dict()["weight"].copy_(q_weight)
                layer.attention.self.mark_proj.state_dict()["weight"].copy_(k_weight)

                log.debug(f"\n******************************")
                log.debug(f"LAYER NUMBER {layer_no}")
                log.debug(f"q_weight == mark_q_weight: {torch.equal(q_weight, layer.attention.self.mark_q_proj.weight)}")
                log.debug(f"k_weight == mark_weight: {torch.equal(k_weight, layer.attention.self.mark_proj.weight)}")
                log.debug(f"******************************")
        # Copying position projections as starting point for mark projections
        elif getattr(hparams, "mark_projection_init", "random") == "position":
            for layer_no, layer in enumerate(self.model.base_model.encoder.layer):
                pos_proj_weight = layer.attention.self.pos_proj.weight.detach()
                pos_q_proj_weight = layer.attention.self.pos_q_proj.weight.detach()
                layer.attention.self.mark_q_proj.state_dict()["weight"].copy_(pos_q_proj_weight)
                layer.attention.self.mark_proj.state_dict()["weight"].copy_(pos_proj_weight)

                log.debug(f"\n******************************")
                log.debug(f"LAYER NUMBER {layer_no}")
                log.debug(f"pos_q_proj_weight == mark_q_proj_weight: {torch.equal(pos_q_proj_weight, layer.attention.self.mark_q_proj.weight)}")
                log.debug(f"pos_proj_weight == mark_proj_weight: {torch.equal(pos_proj_weight, layer.attention.self.mark_proj.weight)}")
                log.debug(f"******************************")

        # Using 1/0 embeddings for mark sequence 
        if getattr(hparams, "mark_embeddings_init", "random") == "binary":
            zeros = torch.zeros(self.config.hidden_size, dtype=torch.float)
            ones = torch.ones(self.config.hidden_size, dtype=torch.float)
            self.model.base_model.encoder.mark_embeddings.state_dict()["weight"].copy_(torch.stack((zeros,ones))) 
        # Using orhthogonal embeddings for mark sequence 
        elif getattr(hparams, "mark_embeddings_init", "random") == "orthogonal":
            u1, v2 = self.model.base_model.encoder.mark_embeddings.weight.detach()
            u1 = u1 / torch.linalg.norm(u1)
            u2 = v2 - (torch.dot(v2,u1) * u1)
            u2 = u2 / torch.linalg.norm(u2)
            assert torch.isclose(torch.dot(u1,u2), torch.tensor(0.0), rtol=1e-05, atol=1e-05), "Gram-Schmidt Implementation Bug"
            self.model.base_model.encoder.mark_embeddings.state_dict()["weight"].copy_(torch.stack((u1,u2))) 

        # If we don't want to update the mark embeddings 
        if getattr(hparams, "mark_embeddings_fixed", False) == True:
            self.model.base_model.encoder.mark_embeddings.weight.requires_grad = False

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
                mark_embeddings_enabled = getattr(self.hparams, "mark_embeddings_enabled", False)
                features = absa_utils.convert_examples_to_features(
                    examples,
                    self.labels,
                    self.hparams.max_seq_length,
                    self.tokenizer,
                    mark_embeddings=mark_embeddings_enabled
                )
                log.debug("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

    def load_dataset(self, mode, batch_size):
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

        if getattr(self.hparams, "mark_embeddings_enabled", False) is True:
            all_pivot_marks = torch.tensor([f.pivot_marks for f in features],
                                                  dtype=torch.long)
            tensors += [all_pivot_marks]
        
        shuffle = mode == 'train'
        return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    @staticmethod
    def map_to_inputs(batch):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                  "labels": batch[3]}
        if len(batch) > 4:
            inputs["marks"] = batch[4]

        return inputs

    def training_step(self, batch, _):
        "Compute loss and log."
        inputs = self.map_to_inputs(batch)
        outputs = self(**inputs)
        loss = outputs[0]
        tensorboard_logs = {'train_loss_step': loss, 'lr': self.lr_scheduler.get_last_lr()[-1]}
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss, 'step': self.current_epoch}
        return {'loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, _):
        "Compute validation."
        inputs = self.map_to_inputs(batch)
        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu()#.numpy()
        target = inputs["labels"].detach().cpu()#.numpy()
        return {"val_loss_step": tmp_eval_loss.detach().cpu(), "pred": preds, "target": target}

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

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()  # By default, PL will only step every epoch.

    def train_dataloader(self):
        train_batch_size = self.hparams.train_batch_size
        dataloader = self.load_dataset("train", train_batch_size)
        gpus = self.hparams.gpus
        num_gpus = len(gpus) if isinstance(gpus, list) else gpus
        t_total = (
            (len(dataloader.dataset) // (train_batch_size * max(1, num_gpus)))
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        return dataloader

    def val_dataloader(self):
        return self.load_dataset("dev", self.hparams.eval_batch_size)

    def test_dataloader(self):
        return self.load_dataset("test", self.hparams.eval_batch_size)

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length)))

    def get_str(self) -> str:
        model_str = f'{self.hparams.model_type}'
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
