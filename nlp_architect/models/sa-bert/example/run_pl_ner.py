import glob
import logging
import os
from time import sleep

from pytorch_lightning.core.saving import load_hparams_from_yaml
import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

from lightning_base import BaseTransformer, generic_trainer
from utils_ner import convert_examples_to_features, get_labels, read_examples_from_file
from argparse import Namespace

logger = logging.getLogger(__name__)


class NERTransformer(BaseTransformer):
    """
    A training module for NER. See BaseTransformer for the core options.
    """

    mode = "token-classification"

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams =  Namespace(**hparams)
        self.labels = get_labels(hparams.labels)
        num_labels = len(self.labels)
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        super().__init__(hparams, num_labels, self.mode)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_num):
        "Compute loss and log."
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if self.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids

        outputs = self(**inputs)
        loss = outputs[0]
        tensorboard_logs = {"loss": loss, "rate": self.lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def prepare_data(self):
        "Called to initialize data. Use the call to construct features"
        args = self.hparams
        for mode in ["train", "dev", "test"]:
            cached_features_file = self._feature_file(mode)
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cached_features_file)
                features = torch.load(cached_features_file)
            else:
                logger.info("Creating features from dataset file at %s", args.data_dir)
                examples = read_examples_from_file(args.data_dir, mode)
                features = convert_examples_to_features(
                    examples,
                    self.labels,
                    args.max_seq_length,
                    self.tokenizer,
                    cls_token_at_end=bool(self.config.model_type in ["xlnet"]),
                    cls_token=self.tokenizer.cls_token,
                    cls_token_segment_id=2 if self.config.model_type in ["xlnet"] else 0,
                    sep_token=self.tokenizer.sep_token,
                    sep_token_extra=False,
                    pad_on_left=bool(self.config.model_type in ["xlnet"]),
                    pad_token=self.tokenizer.pad_token_id,
                    pad_token_segment_id=self.tokenizer.pad_token_type_id,
                    pad_token_label_id=self.pad_token_label_id,
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
            TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids), batch_size=batch_size
        )

    def validation_step(self, batch, batch_idx):
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
        ret, preds, targets = self._eval_end(outputs)
        logs = ret["log"]
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs):
        ret, predictions, targets = self._eval_end(outputs)
        logs = ret["log"]
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}


if __name__ == "__main__":
    config = load_hparams_from_yaml('/home/daniel_nlp/nlp-architect/nlp_architect/models/sa-bert/example.yaml')
    args = Namespace(**config)

    model = NERTransformer(args)
    trainer = generic_trainer(model, args)

    if args.do_train:
        trainer.fit(model)
    trainer.logger.log_hyperparams(args)
    trainer.logger.save()

    if args.do_predict:
        sleep(2)
        print('\n\n\nTESTING: \n\n')
        
        # Bug in pl==0.85 requires test gpu num == 1
        args.gpus = 1
        trainer = generic_trainer(model, args)

        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        print('CHECKPOINT')
        print(checkpoints)
        model = model.load_from_checkpoint(checkpoints[-1])
        model.freeze()
        trainer.test(model)
