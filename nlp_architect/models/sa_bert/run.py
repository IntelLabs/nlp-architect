import glob
import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.core.saving import load_hparams_from_yaml
import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
import absa_utils
from argparse import Namespace
import random
from pathlib import Path
from typing import Any, Dict
from sys import argv

# pylint: disable=attribute-defined-outside-init

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus > 0:
        torch.cuda.manual_seed_all(args.seed)

class BertForToken(pl.LightningModule):
    def __init__(self, hparams, config=None,
                tokenizer=None, model=None, **config_kwargs):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        self.hparams = hparams
        self.labels = absa_utils.get_labels(hparams.labels)
        num_labels = len(self.labels)
        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        self.step_count = 0
        self.tfmr_ckpts = {}
        self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                **({"num_labels": num_labels} if num_labels is not None else {}),
                **config_kwargs,
            )
        else:
            self.config: PretrainedConfig = config
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer: PreTrainedTokenizer = tokenizer
        self.model_type = AutoModelForTokenClassification
        if model is None:
            self.model = self.model_type.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=self.config,
                cache_dir=cache_dir,
            )
        else:
            self.model = model

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, _):
        "Compute loss and log."
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        inputs["token_type_ids"] = batch[2]
        outputs = self(**inputs)
        loss = outputs[0]
        tensorboard_logs = {"loss": loss, "rate": self.lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def prepare_data(self):
        "Called to initialize data. Use the call to construct features"
        args = self.hparams
        for mode in "train", "dev", "test":
            cached_features_file = self._feature_file(mode)
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cached_features_file)
                features = torch.load(cached_features_file)
            else:
                logger.info("Creating features from dataset file at %s", args.data_dir)
                examples = absa_utils.read_examples_from_file(args.data_dir, mode)
                features = absa_utils.convert_examples_to_features(
                    examples,
                    self.labels,
                    args.max_seq_length,
                    self.tokenizer,
                    self.pad_token_label_id
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

    def load_dataset(self, mode, batch_size):
        "Load datasets. Called after prepare data."
        cached_features_file = self._feature_file(mode)
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        if features[0].token_type_ids is not None:
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        else:
            all_token_type_ids = torch.tensor([0 for f in features], dtype=torch.long)
            # HACK(we will not use this anymore soon)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        return DataLoader(
            TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids), batch_size=batch_size, num_workers=self.hparams.num_workers
        )

    def validation_step(self, batch, _):
        "Compute validation"

        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if self.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids
        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()
        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs):
        "Evaluation called for both Val and Test"
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)
        preds = np.argmax(preds, axis=2)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        label_map = {i: label for i, label in enumerate(self.labels)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "val_loss": val_loss_mean,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list

    def validation_epoch_end(self, outputs):
        ret, _, _ = self._eval_end(outputs)
        logs = ret["log"]
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs):
        ret, _, _ = self._eval_end(outputs)
        logs = ret["log"]
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def load_hf_checkpoint(self, *args, **kwargs):
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()  # By default, PL will only step every epoch.
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(self.lr_scheduler.get_lr())}
        self.logger.log_metrics(lrs)

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def train_dataloader(self):
        train_batch_size = self.hparams.train_batch_size
        dataloader = self.load_dataset("train", train_batch_size)

        t_total = (
            (len(dataloader.dataset) // (train_batch_size * max(1, self.hparams.gpus)))
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
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
                str(self.hparams.max_seq_length),
            ),
        )

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_tfmr")
        save_path.mkdir(exist_ok=True)
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.tfmr_ckpts[self.step_count] = save_path

class LoggingCallback(pl.Callback):
    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Test results *****")
        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def generic_trainer(model: BertForToken, args, gpus=None):
    # init model
    set_seed(args)

    Path(model.hparams.output_dir).mkdir(exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1
    )
    gpus = args.gpus if gpus is None else gpus
    distributed_backend = "ddp" if gpus > 1 else None
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    return pl.Trainer(
        logger=True,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gpus=gpus,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
        fast_dev_run=args.fast_dev_run,
        val_check_interval=args.val_check_interval,
        weights_summary=None,
        resume_from_checkpoint=args.resume_from_checkpoint,
        distributed_backend=distributed_backend,
    )

def load_config(name):
    configs_dir = Path(os.path.dirname(os.path.realpath(__file__))) / 'configs'
    config = Namespace(**load_hparams_from_yaml(configs_dir / (name + '.yaml')))
    assert config
    return config

# pylint: disable=no-member
def main(config_yaml):
    config = load_config(config_yaml)
    
    model = BertForToken(config)
    trainer = generic_trainer(model, config)

    if config.do_train:
        trainer.fit(model)
    trainer.logger.log_hyperparams(config)
    trainer.logger.save()

    if config.do_predict:        
        # Bug in pl==0.85 -> testing only works with num gpus=1
        trainer = generic_trainer(model, config, gpus=1)

        checkpoints = list(sorted(glob.glob(os.path.join(config.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1])
        model.freeze()
        trainer.test(model)

if __name__ == "__main__":
    argv = ['', 'example']
    main(argv[1])
