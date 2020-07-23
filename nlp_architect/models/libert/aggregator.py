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
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# pylint: disable=import-error
from tensorflow.core.util.event_pb2 import Event

OUT_BASE = 'aggregate'

def extract(dpath, versions):
    accumulators = [EventAccumulator(str(dpath / dname)).Reload().scalars \
        for dname in versions if dname != OUT_BASE]
    # Filter non event files
    accumulators = [acc for acc in accumulators if acc.Keys()]
    # Get and validate all scalar keys
    all_keys = [tuple(acc.Keys()) for acc in accumulators]
    assert len(set(all_keys)) == 1, "All runs need to have the same scalar keys.\
        There are mismatches in {}".format(all_keys)
    keys = all_keys[0]

    keys = [k for k in keys if 'step' not in k and 'lr' not in k]

    all_events_per_key = [[acc.Items(key) for acc in accumulators] for key in keys]
    # Get and validate all steps per key
    all_steps_per_key = [[tuple(e.step for e in events) for events in all_events]
                         for all_events in all_events_per_key]
    for i, all_steps in enumerate(all_steps_per_key):
        assert len(set(all_steps)) == 1, "For scalar {} the step numbering or count doesn't match.\
            Step count for all runs: {}".format(keys[i], [len(steps) for steps in all_steps])
    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]
    # Get and average wall times per step per key
    wall_times_per_key = [np.mean([tuple(e.wall_time for e in events) for \
        events in all_events], axis=0) for all_events in all_events_per_key]
    # Get values per step per key
    values_per_key = [[[scalar_event.value for scalar_event in events] for events in all_events]
                      for all_events in all_events_per_key]
    all_per_key = dict(zip(keys, zip(steps_per_key, wall_times_per_key, values_per_key)))
    return all_per_key

def aggregate_to_summary(agg_path, aggregation_ops, extracted):
    for op in aggregation_ops:
        aggregations_per_key = {key: (steps, wall_times, op(values, axis=0))\
            for key, (steps, wall_times, values) in extracted.items()}
        write_summary(agg_path / op.__name__, aggregations_per_key)

def write_summary(dpath, aggregations_per_key):
    writer = tf.compat.v1.summary.FileWriter(dpath)
    for key, (steps, wall_times, aggregations) in aggregations_per_key.items():
        for step, wall_time, aggregation in zip(steps, wall_times, aggregations):
            summary = tf.compat.v1.Summary(
                value=[tf.compat.v1.Summary.Value(tag=key, simple_value=aggregation)])
            scalar_event = Event(wall_time=wall_time, step=step, summary=summary)
            writer.add_event(scalar_event)
        writer.flush()

def aggregate_to_csv(agg_path, aggregation_ops, extracted):
    csv_dir = agg_path / 'csv'
    os.makedirs(csv_dir)
    # pylint: disable=abstract-class-instantiated
    with pd.ExcelWriter(agg_path / 'all_metrics.xlsx') as xlsx_writer:
        for key, (steps, _, values) in extracted.items():
            aggregations = [op(values, axis=0) for op in aggregation_ops]
            valid_key = get_valid_filename(key)
            csv_out = csv_dir / (valid_key + '.csv')
            write_csv(csv_out, xlsx_writer, valid_key, aggregations, steps, aggregation_ops)

def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)

def write_csv(csv_out, xlsx_writer, key, aggregations, steps, ops):
    # columns = ['step'] + [op.__name__ for op in ops]
    columns = [op.__name__ for op in ops]
    df = pd.DataFrame(np.transpose(aggregations), index=steps, columns=columns)
    df.to_excel(xlsx_writer, sheet_name=key)
    df.to_csv(csv_out)

def aggregate(root_dir, versions):
    log_root = Path(root_dir)
    aggregation_ops = [np.mean, np.min, np.max, np.median, np.std, np.var]
    extracted = extract(log_root, versions)
    prev_aggs = [d for d in os.listdir(log_root) if d.startswith(OUT_BASE)]
    agg_path = OUT_BASE + '_' + ('0' if not prev_aggs else (str(int(max(prev_aggs)[-1]) + 1)))
    args = log_root / agg_path, aggregation_ops, extracted
    aggregate_to_summary(*args)
    aggregate_to_csv(*args)
