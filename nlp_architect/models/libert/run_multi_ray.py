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
from collections import deque
from subprocess import Popen, STDOUT
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from bert_for_token import BertForToken
from log_aggregator import aggregate
from absa_utils import load_config, set_as_latest, write_summary_tables, pretty_datetime
from significance import significance_from_cfg
from trainer import get_logger, get_trainer, log_model_and_version
from contextlib import redirect_stdout, redirect_stderr
import ray

LIBERT_DIR = Path(realpath(__file__)).parent
LOG_ROOT = LIBERT_DIR / 'logs'
GPUS_LOG = LOG_ROOT / 'gpus'

@ray.remote(max_calls=1)
def run_data(task_idx, cfg_yaml, time, rnd_init, data, log_dir, metric):
    # Routing output to log files
    tasks_log_dir = log_dir / 'tasks'
    with open(tasks_log_dir / f'task_{task_idx}.log', 'a') as log_file:
        with redirect_stdout(log_file):
            with redirect_stderr(log_file):
                cfg = load_config(cfg_yaml)
                cfg.rnd_init = str(rnd_init) == 'True'
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
                    print(f"\n{'*' * 150}\n{' ' * 50}Run {run_i}/{len(runs)}: \
                        {data}, {exper_str}\n{'*' * 150}")
                    exp_id = 'baseline' if model_str == cfg.baseline_str else time

                    if cfg.do_train:
                        trainer = get_trainer(model, data, exper_str, exp_id, log_dir, metric=metric, limit_data=cfg.limit_data)
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
    ray.init()
    
    # Init: config, experiment id, logging
    cfg = load_config(config_yaml)
    time_now = pretty_datetime()
    exp_tag = time_now + cfg.tag
    log_dir = LOG_ROOT / exp_tag
    tasks_log_dir = log_dir / 'tasks'
    os.makedirs(tasks_log_dir, exist_ok=True)
    open(log_dir / 'time.log', 'w').write(f'{time_now}\n')
    set_as_latest(log_dir)
    run_list = product(cfg.base_init, cfg.data)

    args_list = []
    for task_idx, (rnd_init, data) in enumerate(run_list):
        args = task_idx, config_yaml, exp_tag, rnd_init, data, log_dir, cfg.metric
        args_list.append(args)

    # Parallel: each experiment runs on a single GPU
    if cfg.parallel:
        ray_options = {"num_gpus":1}

    # Sequential: each experiment runs in turn on all gpus
    else:
        ray_options = {"num_gpus":len(ray.get_gpu_ids)}

    # Launching Ray tasks
    futures = [run_data.options(**ray_options).remote(*args) for args in args_list]
    results = ray.get(futures)

    for model_str, exp_id in results:
        post_analysis(cfg, log_dir, exp_tag, model_str)

    # Save termination time
    open(log_dir / 'time.log', 'a').write(pretty_datetime())

    ray.shutdown()

def post_analysis(cfg, log_dir, time_tag, model_str):
    # Run significance tests if baseline exists and last run was on model
    if cfg.do_predict and model_str != cfg.baseline_str:
        significance_from_cfg(cfg=cfg, log_dir=log_dir, exp_id=time_tag)

    # Write summary table to CSV
    write_summary_tables(cfg, time_tag)

if __name__ == "__main__":
    if len(argv) == 2:
        main(argv[1])
    else:
        print("Incorrect usage. Please try again.")
