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
# MIT License
# Copyright (c) 2019 Sebastian Penhouet
# GitHub project: https://github.com/Spenhouet/tensorboard-aggregator
# ==============================================================================
"""Aggregates multiple tensorbaord runs."""

import os
import re

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.core.util.event_pb2 import Event
from pathlib import Path

FOLDER_NAME = 'aggregates'


def extract(dpath, subpath):
    scalar_accumulators = [EventAccumulator(str(dpath / dname / subpath)).Reload(
    ).scalars for dname in os.listdir(dpath) if dname != FOLDER_NAME]

    # Filter non event files
    scalar_accumulators = [scalar_accumulator for scalar_accumulator in scalar_accumulators if scalar_accumulator.Keys()]

    # Get and validate all scalar keys
    all_keys = [tuple(scalar_accumulator.Keys()) for scalar_accumulator in scalar_accumulators]
    assert len(set(all_keys)) == 1, "All runs need to have the same scalar keys. There are mismatches in {}".format(all_keys)
    keys = all_keys[0]

    all_scalar_events_per_key = [[scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators] for key in keys]

    # Get and validate all steps per key
    all_steps_per_key = [[tuple(scalar_event.step for scalar_event in scalar_events) for scalar_events in all_scalar_events]
                         for all_scalar_events in all_scalar_events_per_key]

    for i, all_steps in enumerate(all_steps_per_key):
        assert len(set(all_steps)) == 1, "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
            keys[i], [len(steps) for steps in all_steps])

    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

    # Get and average wall times per step per key
    wall_times_per_key = [np.mean([tuple(scalar_event.wall_time for scalar_event in scalar_events) for scalar_events in all_scalar_events], axis=0)
                          for all_scalar_events in all_scalar_events_per_key]

    # Get values per step per key
    values_per_key = [[[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
                      for all_scalar_events in all_scalar_events_per_key]

    all_per_key = dict(zip(keys, zip(steps_per_key, wall_times_per_key, values_per_key)))

    return all_per_key

def aggregate_to_summary(dpath, aggregation_ops, extracts_per_subpath):
    for op in aggregation_ops:
        for subpath, all_per_key in extracts_per_subpath.items():
            path = dpath / FOLDER_NAME / op.__name__ / dpath.name / subpath
            aggregations_per_key = {key: (steps, wall_times, op(values, axis=0)) for key, (steps, wall_times, values) in all_per_key.items()}
            write_summary(path, aggregations_per_key)

def write_summary(dpath, aggregations_per_key):
    writer = tf.summary.FileWriter(dpath)

    for key, (steps, wall_times, aggregations) in aggregations_per_key.items():
        for step, wall_time, aggregation in zip(steps, wall_times, aggregations):
            summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=aggregation)])
            scalar_event = Event(wall_time=wall_time, step=step, summary=summary)
            writer.add_event(scalar_event)

        writer.flush()

def aggregate_to_csv(dpath, aggregation_ops, extracts_per_subpath):
    for subpath, all_per_key in extracts_per_subpath.items():
        for key, (steps, wall_times, values) in all_per_key.items():
            aggregations = [op(values, axis=0) for op in aggregation_ops]
            write_csv(dpath, subpath, key, dpath.name, aggregations, steps, aggregation_ops)

def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)

def write_csv(dpath, subpath, key, fname, aggregations, steps, aggregation_ops):
    path = dpath / FOLDER_NAME
    if not path.exists():
        os.makedirs(path)
    file_name = get_valid_filename(key) + '-' + get_valid_filename(subpath) + '-' + fname + '.csv'
    aggregation_ops_names = [aggregation_op.__name__ for aggregation_op in aggregation_ops]
    df = pd.DataFrame(np.transpose(aggregations), index=steps, columns=aggregation_ops_names)
    df.to_csv(path / file_name, sep=';')


def aggregate(root_dir, subpaths):
    dpath = Path(root_dir)
    name = dpath.name
    subpath_names = [dpath / subpath for subpath in subpaths]
    aggregation_ops = [np.mean, np.min, np.max, np.median, np.std, np.var]
    print("Started aggregation {}".format(name))
    extracts_per_subpath = {subpath: extract(dpath, subpath) for subpath in subpath_names}
    args = dpath, aggregation_ops, extracts_per_subpath
    aggregate_to_summary(*args)
    # aggregate_to_csv(*args)
    print("Ended aggregation {}".format(name))


# if __name__ == '__main__':
#     def param_list(param):
#         p_list = ast.literal_eval(param)
#         if type(p_list) is not list:
#             raise argparse.ArgumentTypeError("Parameter {} is not a list".format(param))
#         return p_list

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--path", type=str, help="main path for tensorboard files", default=os.getcwd())
#     parser.add_argument("--subpaths", type=param_list, help="subpath sturctures", default=['test', 'train'])
#     parser.add_argument("--output", type=str, help="aggregation can be saves as tensorboard file (summary) or as table (csv)", default='summary')

#     args = parser.parse_args()

#     path = Path(args.path)

#     if not path.exists():
#         raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(path))

#     subpaths = [path / dname / subpath for subpath in args.subpaths for dname in os.listdir(path) if dname != FOLDER_NAME]

#     for subpath in subpaths:
#         if not os.path.exists(subpath):
#             raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(subpath))

#     if args.output not in ['summary', 'csv']:
#         raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(args.output))

#     aggregate(path, args.output, args.subpaths)