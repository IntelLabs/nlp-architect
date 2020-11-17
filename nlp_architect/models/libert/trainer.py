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
# pylint: disable=logging-fstring-interpolation, no-member

import logging
from pathlib import Path
from os.path import realpath

import pytorch_lightning as pl
from bert_for_token import LoggingCallback

logging.getLogger("transformers").setLevel('ERROR')
logging.getLogger("pytorch_lightning").setLevel('WARNING')
LIBERT_DIR = Path(realpath(__file__)).parent

def log_model_and_version(trainer, cfg, versions, save=True):
    logger = trainer.logger
    logger.log_hyperparams(cfg)
    if save:
        logger.save()
    versions.append(logger.experiment.log_dir)

def get_logger(data, experiment, exp_id, log_dir=None, suffix=None):
    suffix = '_' + suffix if suffix else ''
    save_dir = log_dir if log_dir else LIBERT_DIR / 'logs'
    return pl.loggers.TestTubeLogger(save_dir= Path(save_dir) / data, 
                                     name=experiment + suffix, version=exp_id)

def get_trainer(model, data, experiment, exp_id, log_dir=None, gpus=None, metric='micro_f1', limit_data=1.0):
    """Init trainer for model training/testing."""
    Path(model.hparams.output_dir).mkdir(exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=str(model.hparams.output_dir) + "/_{epoch}-{" + metric + ":.4f}",
        prefix='_'.join([data, experiment, exp_id]), monitor=metric, mode="max",
        save_top_k=1, save_weights_only=True
    )
    logger = get_logger(data, experiment, exp_id, log_dir, suffix='train')
    gpus = model.hparams.gpus if gpus is None else gpus
    num_gpus = len(gpus) if isinstance(gpus, list) else gpus
    return pl.Trainer(
        logger=logger,
        log_save_interval=10,
        row_log_interval=10,
        accumulate_grad_batches=model.hparams.accumulate_grad_batches,
        gpus=gpus,
        max_epochs=model.hparams.max_epochs,
        gradient_clip_val=model.hparams.gradient_clip_val,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
        fast_dev_run=model.hparams.fast_dev_run,
        val_check_interval=model.hparams.val_check_interval,
        weights_summary=None,
        resume_from_checkpoint=model.hparams.resume_from_checkpoint,
        distributed_backend="ddp" if num_gpus > 1 else None,
        benchmark=True,
        deterministic=True,
        limit_train_batches=limit_data,
        limit_val_batches=limit_data,
        limit_test_batches=limit_data,
        precision=16
    )
