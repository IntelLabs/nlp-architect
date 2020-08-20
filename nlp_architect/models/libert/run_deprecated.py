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
import sys
from pathlib import Path
from os.path import realpath
from itertools import product
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from bert_for_token import BertForToken
from log_aggregator import aggregate
from absa_utils import load_config
from significance import significance_from_cfg
from trainer import get_logger, get_trainer, log_model_and_version

LIBERT_DIR = Path(realpath(__file__)).parent

def main(config_yaml):
    cfg = load_config(config_yaml)
    timed_id = datetime.now().strftime("%a_%b_%d_%H:%M:%S") + cfg.tag
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
        significance_from_cfg(cfg, LIBERT_DIR / 'logs', exp_id)

if __name__ == "__main__":
    main(sys.argv[1])