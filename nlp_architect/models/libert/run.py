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
from aggregator import aggregate
from sys import argv
from os.path import dirname, realpath
from pathlib import Path
from itertools import product
from pytorch_lightning.loggers import TensorBoardLogger
import logging
log = logging.getLogger(__name__)

def get_trainer(model, data, experiment, gpus_override=None):
    """Init trainer for model training/testing."""
    Path(model.hparams.output_dir).mkdir(exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=model.hparams.output_dir, prefix="chjeckpoint", monitor="val_loss",
        mode="min", save_top_k=1
    )
    gpus = model.hparams.gpus if gpus_override is None else gpus_override
    distributed_backend = "ddp" if gpus > 1 else None

    logger = TensorBoardLogger(
        save_dir=Path(dirname(realpath(__file__))) / 'logs' / data,
        name=experiment
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
        profiler=True,
        benchmark=True,
        deterministic=False
    )

# pylint: disable=no-member
def main(config_yaml):
    config = load_config(config_yaml)

    versions = []
    runs = list(product(config.seeds, config.splits))
    for i, (seed, split) in enumerate(runs):
        experiment = "seed_{}_split_{}".format(seed, split)
        log.info('\n{}\n{}Run {}/{}: {}, {}\n{}'\
            .format('*' * 150, ' ' * 50, i + 1, len(runs), config.data, experiment, '*' * 150))

        pl.seed_everything(seed)
        config.data_dir = config.data + '_' + str(split)
        model = BertForToken(config)

        if config.do_train:
            trainer = get_trainer(model, config.data, experiment)
            trainer.fit(model)

            trainer.logger.log_hyperparams(config)
            trainer.logger.save()
            versions.append(Path(trainer.logger.log_dir))

        if config.do_predict:        
            # Bug in pytorch_lightning==0.85 -> testing only works with num gpus=1
            trainer = get_trainer(model, config.data, experiment + '_test', gpus_override=1)
            trainer.test(load_last_ckpt(model))

    if len(versions) > 1:
        aggregate(versions)

if __name__ == "__main__":
    argv = ['', '']
    # argv[1] = 'example'
    argv[1] = 'sanity'

    main(argv[1])
