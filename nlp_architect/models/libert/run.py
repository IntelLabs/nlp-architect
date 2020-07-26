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
import pytorch_lightning as pl
from bert_for_token import BertForToken, load_config, load_last_ckpt, LoggingCallback
from log_aggregator import aggregate
from sys import argv
from pathlib import Path
from os.path import realpath
from itertools import product
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import _logger as log

LIBERT_OUT = Path(realpath(__file__)).parent / 'out'

def get_trainer(model, data, experiment, version=None, gpus_override=None):
    """Init trainer for model training/testing."""
    Path(model.hparams.output_dir).mkdir(exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=model.hparams.output_dir, prefix="checkpoint", monitor="val_loss",
        mode="min", save_top_k=1
    )
    gpus = model.hparams.gpus if gpus_override is None else gpus_override
    distributed_backend = "ddp" if gpus > 1 else None

    logger = TensorBoardLogger(
        save_dir= LIBERT_OUT / 'logs' / data,
        name=experiment,
        version=version
    )

    return pl.Trainer(
        logger=logger,
        log_save_interval=10,
        row_log_interval=10,
        accumulate_grad_batches=model.hparams.accumulate_grad_batches,
        gpus=gpus,
        max_epochs=model.hparams.max_epochs,
        gradient_clip_val=model.hparams.gradient_clip_val,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback(), pl.callbacks.LearningRateLogger()],
        fast_dev_run=model.hparams.fast_dev_run,
        val_check_interval=model.hparams.val_check_interval,
        weights_summary=None,
        resume_from_checkpoint=model.hparams.resume_from_checkpoint,
        distributed_backend=distributed_backend,
        # profiler=True,
        benchmark=True,
        deterministic=True
    )

# pylint: disable=no-member
def main(config_yaml):
    cfg = load_config(config_yaml)

    versions = []
    runs = list(product(cfg.seeds, cfg.splits))
    for i, (seed, split) in enumerate(runs):
        random_init_str = 'random_init_' if cfg.random_init and cfg.model_type == 'libert' else ''
        experiment = "{}_{}seed_{}_split_{}".format(cfg.model_type, random_init_str, seed, split)
        log.info('\n{}\n{}Run {}/{}: {}, {}\n{}'\
            .format('*' * 150, ' ' * 50, i + 1, len(runs), cfg.data, experiment, '*' * 150))

        pl.seed_everything(seed)
        cfg.data_dir = cfg.data + '_' + str(split)
        model = BertForToken(cfg)

        if cfg.do_train:
            trainer = get_trainer(model, cfg.data, experiment, cfg.version)
            trainer.fit(model)

            trainer.logger.log_hyperparams(cfg)
            trainer.logger.save()
            versions.append(Path(trainer.logger.log_dir))

        if cfg.do_predict:      
            # Bug in pytorch_lightning==0.85 -> testing only works with num gpus=1
            trainer = get_trainer(model, cfg.data, experiment + '_test', gpus_override=1)
            trainer.test(load_last_ckpt(model))

    # Aggregate tensorboard log metrics for all runs
    if len(versions) > 1:
        aggregate(versions)

if __name__ == "__main__":
    argv = ['', '']
    # argv[1] = 'example'
    argv[1] = 'sanity'

    main(argv[1])
