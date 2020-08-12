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
# pylint: disable=logging-fstring-interpolation, no-member, unsubscriptable-object
# pylint: disable=no-value-for-parameter

import os
from pathlib import Path
from sys import argv, executable as python
from os.path import realpath
from itertools import product
from datetime import datetime as dt
from collections import deque
from subprocess import Popen, STDOUT
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from bert_for_token import BertForToken
from log_aggregator import aggregate
from absa_utils import load_config, set_as_latest_log_dir
from significance import significance_from_cfg
from trainer import get_logger, get_trainer, log_model_and_version

LIBERT_DIR = Path(realpath(__file__)).parent
LOG_ROOT = LIBERT_DIR / 'logs'

def run_data(cfg_yaml, time, rnd_init, data, log_dir, metric):
    cfg = load_config(cfg_yaml)
    cfg.rnd_init = rnd_init == 'True'
    cfg.gpus = 1
    train_versions, test_versions = [], []
    runs = list(product(cfg.seeds, cfg.splits))
    for run_i, (seed, split) in enumerate(runs, start=1):
        pl.seed_everything(seed)
        cfg.data_dir = f'{data}_{split}'
        model = BertForToken(cfg)
        model_str = f'{cfg.model_type}_rnd_init' if cfg.rnd_init else f'{cfg.model_type}'
        exper_str = f'{model_str}_seed_{seed}_split_{split}'
        log.info(f"\n{'*' * 150}\n{' ' * 50}Run {run_i}/{len(runs)}: \
            {data}, {exper_str}\n{'*' * 150}")
        exp_id = 'baseline' if model_str == cfg.baseline_str else time

        if cfg.do_train:
            trainer = get_trainer(model, data, exper_str, exp_id, log_dir, metric=metric)
            trainer.fit(model)
            log_model_and_version(trainer, cfg, train_versions)

        if cfg.do_predict:
            # Switch to test logger
            trainer.logger = get_logger(data, exper_str, exp_id, log_dir, suffix='test')
            trainer.test()
            log_model_and_version(trainer, cfg, test_versions, save=False)

    # Aggregate tensorboard log metrics for all runs on this data
    if len(train_versions) > 1:
        aggregate(train_versions, exp_id + '_train', model_str)
        aggregate(test_versions, exp_id + '_test', model_str)
    return model_str, exp_id


def main(config_yaml):
    cfg = load_config(config_yaml)
    os.makedirs(LOG_ROOT, exist_ok=True)

    time_tag = dt.now().strftime("%a_%b_%d_%H:%M:%S") + cfg.tag
    open(LOG_ROOT / 'time.log', 'w').write(f'{time_tag}\n')

    log_dir = LOG_ROOT / time_tag

    run_queue = deque(product(cfg.base_init, cfg.data))
    num_procs = min(len(cfg.gpus), len(run_queue))

    this_module = Path(realpath(__file__))
    while run_queue:
        procs = []
        for gpu_i in cfg.gpus[:num_procs]:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_i)
            rnd_init, data = run_queue.popleft()
            args = this_module, config_yaml, time_tag, rnd_init, data, log_dir, cfg.metric
            cmd = [python] + [f'{_}' for _ in args]
            with open(LOG_ROOT / f'gpu_{gpu_i}.log', 'a') as log_file:
                log_file.truncate(0)
                proc = Popen(cmd, bufsize=-1, stdout=log_file, stderr=STDOUT)

            print(f"PID: {proc.pid}: yaml: {config_yaml}.yaml, time: {time_tag}, " \
                f"rnd_init: {rnd_init}, data: {data}, gpu: {gpu_i}")
            procs.append(proc)
            model_str = f'{cfg.model_type}_rnd_init' if rnd_init else f'{cfg.model_type}'
            if not run_queue:
                break

        for proc in procs:
            print(f'waiting for pid {proc.pid}')
            proc.wait()

    # Run significance tests if baseline exists and last run was on model
    if model_str != cfg.baseline_str:
        significance_from_cfg(cfg=cfg, log_dir=log_dir, exp_id=time_tag)
    
    open(LOG_ROOT / 'time.log', 'a').write(dt.now().strftime("%a_%b_%d_%H:%M:%S"))

    set_as_latest_log_dir(log_dir)

if __name__ == "__main__":
    if len(argv) == 2:
        main(argv[1])
    else:
        run_data(*argv[1:])
