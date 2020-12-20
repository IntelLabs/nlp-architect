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
#%%
import os
import csv
from pathlib import Path
from argparse import Namespace
from sys import argv
from os.path import realpath
from itertools import product
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from bert_for_token import BertForToken
from log_aggregator import aggregate
from absa_utils import copy_config, read_config, prepare_config, set_as_latest, write_summary_tables, pretty_datetime, prepare_pattern_info_in_cfg
from significance import significance_from_cfg
from trainer import get_logger, get_trainer, log_model_and_version
from contextlib import redirect_stdout, redirect_stderr
from torch.cuda import device_count
import ray

LIBERT_DIR = Path(realpath(__file__)).parent
LOG_ROOT = LIBERT_DIR / 'logs'
GPUS_LOG = LOG_ROOT / 'gpus'

@ray.remote(max_calls=1)
def run_data(task_idx, cfg_orig, time, baseline, data, log_dir, metric):
    # Routing output to log files
    tasks_log_dir = log_dir / 'tasks'
    with open(tasks_log_dir / f'task_{task_idx}.log', 'a', encoding='utf-8') as log_file:
        with redirect_stdout(log_file):
            with redirect_stderr(log_file):
                cfg = Namespace(**vars(cfg_orig)) # copy cfg into local cfg
                cfg.baseline = str(baseline) == 'True'
                cfg.gpus = 1

                train_versions, test_versions = [], []

                runs = list(product(cfg.seeds, cfg.splits(data)))
                for run_i, (seed, split) in enumerate(runs, start=1):
                    pl.seed_everything(seed)
                    if cfg.is_cross_domain(data):
                        cfg.data_dir = f'{data}_{split}'
                    else:               # in-domain setting - stating only domain name 
                        cfg.data_dir = f'{data}_in_domain_{split}'

                    # pattern information (i.e. set of pattern classes) is dependent on dataset & split
                    prepare_pattern_info_in_cfg(cfg)
                    
                    model = BertForToken(cfg)
                    model_str = f'{cfg.model_type}_baseline' if cfg.baseline else f'{cfg.model_type}'
                    exper_str = f'{model_str}_seed_{seed}_split_{split}'
                    log.info(f"\n{'*' * 150}\n{' ' * 50}Run {run_i}/{len(runs)}: \
                        {data}, {exper_str}\n{'*' * 150}")
                    print(f"\n{'*' * 150}\n{' ' * 50}Run {run_i}/{len(runs)}: \
                        {data}, {exper_str}\n{'*' * 150}")
                    exp_id = f'{cfg.model_type}_baseline' if cfg.baseline else cfg.model_type

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
                if len(train_versions) >= 1:
                    aggregate(train_versions, exp_id + '_train', model_str)
                    aggregate(test_versions, exp_id + '_test', model_str)
                return model_str, exp_id

def main(config_yaml):
    # Init: config, experiment id, logging
    cfg = read_config(config_yaml)
    time_start = pretty_datetime()
    exp_id = time_start + cfg.tag
    base_log_dir = LOG_ROOT / exp_id
    print(f"\nStarting experiment - logging at 'logs/{exp_id}' \n")
    copy_config(config_yaml, base_log_dir)  # save a copy of config in log dir - for easier tracking
    
    # run experiments for several formalisms - each log in a subdir  
    for formalism in cfg.formalisms.split():
        cfg.formalism = formalism.lower()
        prepare_config(cfg)

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
            ray.init(num_gpus=num_gpus_req)
        
        # Sequential: run each experiemnt in turn (on all gpus)
        else:
            ray.init(num_gpus=num_gpus_req, local_mode=True)

        # Setting up log dir
        log_dir = base_log_dir / formalism
        tasks_log_dir = log_dir / 'tasks'
        os.makedirs(tasks_log_dir, exist_ok=True)
        open(log_dir / 'time.log', 'w', encoding='utf-8').write(f'{pretty_datetime()}\n')
        set_as_latest(log_dir)

        # Setting up run configurations
        run_list = product(cfg.baseline, cfg.data)
        args_list = []
        for task_idx, (baseline, data) in enumerate(run_list):
            args = task_idx, cfg, exp_id, baseline, data, log_dir, cfg.metric
            args_list.append(args)

        # Launching Ray tasks
        futures = [run_data.options(num_gpus=1).remote(*args) for args in args_list]
        ray.get(futures)

        post_analysis(cfg, log_dir, exp_id)

        # Save termination time
        open(log_dir / 'time.log', 'a', encoding='utf-8').write(pretty_datetime())

        ray.shutdown()
    
        print(f"\n\n Experiment Done for {formalism}. logs are at {log_dir.absolute()}\n\n" + "~"*20 + "\n\n")
    # General (cross-formalisms) finalization 
    prepare_all_formalisms_summary_table(exp_id)
    # print Full Summary table to stdout
    with open(LOG_ROOT / exp_id / "Full-Summary.csv", "r", encoding="utf-8") as f:
        print(f.read())
    print("\nDone all experiments.")

    
    
def post_analysis(cfg, log_dir, exp_id):
    # Run significance tests if baseline exists and last run was on model
    if cfg.do_predict and True in cfg.baseline and False in cfg.baseline:
        sig_result = significance_from_cfg(cfg=cfg, log_dir=log_dir, exp_id=exp_id)

        # Write summary table to CSV
        write_summary_tables(cfg, exp_id, log_dir, sig_result)

def prepare_all_formalisms_summary_table(exp_id: str):
    with open(LOG_ROOT / exp_id / "Full-Summary.csv", "w", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        for frlsm_dir in (LOG_ROOT / exp_id).iterdir():
            if frlsm_dir.is_dir():
                writer.writerows([[frlsm_dir.name.upper()], []])
                with open(frlsm_dir / f"{exp_id}.csv", "r", encoding="utf-8") as fin:
                    reader = csv.reader(fin)
                    frlsm_rows = list(reader)
                writer.writerows(frlsm_rows)
                writer.writerows([[], [], ["~~~~~"]*12, []])    

#%%
if __name__ == "__main__":
    if len(argv) == 2:
        main(argv[1])
    else:
        print("Incorrect usage. Please try again.")


