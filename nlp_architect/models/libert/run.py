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
# pylint: disable=logging-fstring-interpolation

import logging
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from bert_for_token import BertForToken, LoggingCallback
from log_aggregator import aggregate
from sys import argv
from pathlib import Path
from os.path import realpath
from itertools import product
from absa_utils import load_config
from significance import significance_report_from_cfg
from datetime import datetime

logging.getLogger("transformers").setLevel('ERROR')
logging.getLogger("pytorch_lightning").setLevel('WARNING')
LIBERT_DIR = Path(realpath(__file__)).parent

def log_model_and_version(trainer, cfg, versions, save=True):
    logger = trainer.logger
    logger.log_hyperparams(cfg)
    if save:
        logger.save()
    versions.append(logger.experiment.log_dir)

def get_logger(data, experiment, exp_id, suffix=None):
    suffix = '_' + suffix if suffix else ''
    return pl.loggers.TestTubeLogger(save_dir=LIBERT_DIR / 'logs' / data, 
                                     name=experiment + suffix, version=exp_id)

def get_trainer(model, data, experiment, exp_id, gpus=None):
    """Init trainer for model training/testing."""
    Path(model.hparams.output_dir).mkdir(exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=str(model.hparams.output_dir) + "/_{epoch}-{micro_f1:.4f}",
        prefix=experiment + '_' + exp_id, monitor="micro_f1", mode="max", save_top_k=1
    )
    logger = get_logger(data, experiment, exp_id, suffix='train')
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
    )

# pylint: disable=no-member
def main(config_yaml):
    cfg = load_config(config_yaml)
    timed_id = datetime.now().strftime("%a_%b_%d_%H:%M:%S") + cfg.version_tag
    runs = product(cfg.base_init, cfg.data, cfg.seeds, cfg.splits)
    runs_per_data = len(cfg.seeds) * len(cfg.splits)

    for run_i, (base_init, data, seed, split) in enumerate(runs, start=1):
        pl.seed_everything(seed)

        cfg.rnd_init = base_init
        cfg.data_dir = f'{data}_{split}'
        model = BertForToken(cfg)

        model_str = model.get_str()
        exper_str = f'{model_str}_seed_{seed}_split_{split}'
        log.info(f"\n{'*' * 150}\n{' ' * 50}Run {run_i}/{len(runs)}: {data}, {exper_str}\n{'*' * 150}")
        exp_id = 'baseline' if model_str == cfg.baseline_str else timed_id

        if run_i % runs_per_data == 0:
            train_versions, test_versions = [], []

        if cfg.do_train:
            trainer = get_trainer(model, data, exper_str, exp_id)
            trainer.fit(model)
            log_model_and_version(trainer, cfg, train_versions)

        if cfg.do_predict:
            # Switch to test logger
            trainer.logger = get_logger(data, exper_str, exp_id, suffix='test')
            trainer.test()
            log_model_and_version(trainer, cfg, test_versions, save=False)

        # Aggregate tensorboard log metrics for all runs on this data
        if (run_i + 1) % runs_per_data == 0 and len(train_versions) > 1:
            aggregate(train_versions, exp_id + '_train', model_str)
            aggregate(test_versions, exp_id + '_test', model_str)

    if model_str != cfg.baseline_str and 'sanity' not in cfg.data:
        # Print significance report of model results
        significance_report_from_cfg(cfg, LIBERT_DIR / 'logs', exp_id)

if __name__ == "__main__":
    main(argv[1])
