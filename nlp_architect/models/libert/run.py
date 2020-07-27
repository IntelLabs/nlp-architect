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
import logging
import pytorch_lightning as pl
from bert_for_token import BertForToken, LoggingCallback
from log_aggregator import aggregate
from sys import argv
from pathlib import Path
from os.path import realpath
from itertools import product
from absa_utils import load_config, load_last_ckpt, run_log_msg
from significance import significance_report

logging.getLogger("transformers").setLevel('ERROR')
logging.getLogger("pytorch_lightning").setLevel('WARNING')
LIBERT_OUT = Path(realpath(__file__)).parent / 'out'

def get_trainer(model, data, experiment, version=None, gpus=None):
    """Init trainer for model training/testing."""
    Path(model.hparams.output_dir).mkdir(exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=model.hparams.output_dir, prefix="checkpoint", monitor="micro_f1",
        mode="max", save_top_k=1
    )
    gpus = model.hparams.gpus if gpus is None else gpus
    backend = "ddp" if gpus > 1 else None

    logger = pl.loggers.TestTubeLogger(save_dir=LIBERT_OUT / 'logs' / data, name=experiment,
                                       version=version)

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
        distributed_backend=backend,
        benchmark=True,
        deterministic=True,
        # profiler=True
    )

# pylint: disable=no-member
def main(config_yaml):
    cfg = load_config(config_yaml)

    run_i = 1
    for data in cfg.data:
        versions = []
        for seed, split in product(cfg.seeds, cfg.splits):
            pl.seed_everything(seed)

            cfg.data_dir = f'{data}_{split}'
            model = BertForToken(cfg)

            model_str = model.get_str()
            experiment = run_log_msg(cfg, model_str, data, seed, split, run_i)

            if cfg.do_train:
                trainer = get_trainer(model, data, experiment, cfg.version)
                trainer.fit(model)

                tr_logger = trainer.logger
                tr_logger.log_hyperparams(cfg)
                tr_logger.save()
                versions.append(tr_logger.experiment.log_dir)

            if cfg.do_predict:
                # Bug in pytorch_lightning==0.85 -> testing only works with num gpus=1
                trainer = get_trainer(model, data, f'{experiment}_test', gpus=1)
                trainer.test(load_last_ckpt(model))
            run_i += 1

        # Aggregate tensorboard log metrics for all runs on this data
        if len(versions) > 1:
            aggregate(versions)

    if model_str != cfg.baseline_str and 'sanity' not in cfg.data:
        # Print significance report of model results
        master_version = Path(tr_logger.experiment.log_dir).parent.name
        significance_report(cfg.data, master_version, cfg.seeds, cfg.splits, LIBERT_OUT / 'logs',
                        cfg.model_type, cfg.baseline_str, epochs=cfg.max_epochs)

if __name__ == "__main__":
    argv = ['', '']
    # argv[1] = 'example'
    argv[1] = 'sanity'
    main(argv[1])
