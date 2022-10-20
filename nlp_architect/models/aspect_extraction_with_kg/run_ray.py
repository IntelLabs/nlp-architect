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
from sys import argv
from os.path import realpath
from itertools import product
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from deberta_for_token import DebertaForToken
from log_aggregator import aggregate
from absa_utils import load_config, set_as_latest, write_summary_tables, pretty_datetime
from significance import significance_from_cfg
from trainer import get_logger, get_trainer, log_model_and_version
from contextlib import redirect_stdout, redirect_stderr
from torch.cuda import device_count
import ray
import pandas as pd

LIBERT_DIR = Path(realpath(__file__)).parent
LOG_ROOT = LIBERT_DIR / 'logs'
GPUS_LOG = LOG_ROOT / 'gpus'

@ray.remote(num_gpus=1, max_calls=1)
def run_data(task_idx, cfg_yaml, time, rnd_init, data, log_dir, metric):
    # Routing output to log files
    tasks_log_dir = log_dir / 'tasks'
    with open(tasks_log_dir / f'task_{task_idx}.log', 'a', encoding='utf-8') as log_file:
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
                    model = DebertaForToken(cfg)
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
    # Init: config, experiment id, logging
    cfg = load_config(config_yaml)

    # Determining num gpus requested 
    if isinstance(cfg.gpus, list):
        num_gpus_req = len(cfg.gpus)
        cfg.gpus = [str(gpu_id) for gpu_id in cfg.gpus]
        gpu_id_list = ",".join(cfg.gpus)
    elif isinstance(cfg.gpus, int):
        num_gpus_req = cfg.gpus
        cfg.gpus = [str(gpu_id) for gpu_id in range(cfg.gpus)]
        gpu_id_list = ",".join(cfg.gpus)
    else:
        print("Unrecognized GPU configuration.")
        exit()

    # Making sure there are enough GPUs on the system
    num_gpus_avail = device_count()
    assert num_gpus_req <= num_gpus_avail, f"Requested {num_gpus_req} GPUs, only {num_gpus_avail} available."

    # Setting GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_list

    # Parallel: each experiment runs on a single GPU
    if cfg.parallel:
        ray.init(num_gpus=num_gpus_req, dashboard_host="0.0.0.0")
    
    # Sequential: run each experiemnt in turn (on all gpus)
    else:
        ray.init(num_gpus=num_gpus_req, dashboard_host="0.0.0.0", local_mode=True)

    # Setting up log dir
    time_now = pretty_datetime()
    exp_id = time_now + cfg.tag
    log_dir = LOG_ROOT / exp_id
    tasks_log_dir = log_dir / 'tasks'
    os.makedirs(tasks_log_dir, exist_ok=True)
    open(log_dir / 'time.log', 'w', encoding='utf-8').write(f'{time_now}\n')
    set_as_latest(log_dir)

    # Setting up run configurations
    run_list = product(cfg.base_init, cfg.data)
    args_list = []
    for task_idx, (rnd_init, data) in enumerate(run_list):
        args = task_idx, config_yaml, exp_id, rnd_init, data, log_dir, cfg.metric
        args_list.append(args)

    # Launching Ray tasks
    futures = [run_data.remote(*args) for args in args_list]
    ray.get(futures)

    #post_analysis(cfg, log_dir, exp_id)
    results = summarize_results(log_dir)
    results.to_csv(log_dir / 'results.csv')
    print(results[['dataset','asp_f1']].groupby('dataset').mean().reset_index())

    # Save termination time
    open(log_dir / 'time.log', 'a', encoding='utf-8').write(pretty_datetime())

    ray.shutdown()

def post_analysis(cfg, log_dir, exp_id):
    # Run significance tests if baseline exists and last run was on model
    if cfg.do_predict and True in cfg.base_init and False in cfg.base_init:
        significance_from_cfg(cfg=cfg, log_dir=log_dir, exp_id=exp_id)

    # Write summary table to CSV
    write_summary_tables(cfg, exp_id)

def summarize_results(directory):
    folders = [f for f in os.listdir(directory) if f not in ['gpus', 'time.log', 'tasks']]
    results = {}
    for dataset in folders:
        subfolders = os.listdir(directory / dataset)
        subfolders = [m for m in subfolders if m[-4:] == 'test' and 'AGGREGATED' not in m]
        dataset_results = []
        for model in subfolders:
            modelpath = directory / dataset / model
            modelpath = modelpath / os.listdir(modelpath)[0]
            model_results = pd.read_csv(modelpath / 'metrics.csv')
            model_results = model_results.assign(model = model)
            dataset_results.append(model_results)
        dataset_results = pd.concat(dataset_results)
        dataset_results = dataset_results.assign(dataset = dataset)
        results[dataset] = dataset_results
    results = pd.concat(results)
    return results

if __name__ == "__main__":
    if len(argv) == 2:
        main(argv[1])
    else:
        print("Incorrect usage. Please try again.")
