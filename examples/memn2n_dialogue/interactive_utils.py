#!/usr/bin/env python
# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
from builtins import input as text_input
from copy import copy
from functools import reduce

import numpy as np


def interactive_loop(model, babi):
    """
    Loop used to interact with trained conversational agent with access to knowledge base API
    """
    context = []
    response = None
    time_feat = 1
    interactive_output = None
    db_results = []
    allow_oov = False

    print("Building database...")
    print_help()
    db, names_to_idxs, kb_text = build_kb_db(babi)

    while True:
        line_in = text_input('>>> ').strip().lower()
        if not line_in:
            line_in = "<SILENCE>"
        if line_in in ('exit', 'quit'):
            break
        if line_in == 'help':
            print_help()
            continue
        if line_in in ('restart', 'clear'):
            context = []
            response = None
            time_feat = 1
            interactive_output = None
            db_results = []
            print("Memory cleared")
            continue
        if line_in == 'vocab':
            print_human_vocab(babi)
            continue
        if line_in == 'show_memory':
            print_memory(context)
            continue
        if line_in == 'allow_oov':
            allow_oov = not allow_oov
            print("Allow OOV = {}".format(allow_oov))
            continue
        if 'api_call' in line_in:
            db_results = issue_api_call(
                line_in, db, names_to_idxs, kb_text, babi)

        old_context = copy(context)
        user_utt, context, memory, cands_mat, time_feat = babi.process_interactive(
            line_in, context, response, db_results, time_feat)

        if babi.word_to_index['<OOV>'] in user_utt and allow_oov is False:
            oov_word = line_in.split(' ')[
                list(user_utt).index(
                    babi.word_to_index['<OOV>'])]
            print("Sorry, \"{}\" is outside my vocabulary. ".format(oov_word)
                  + "Please say 'allow_oov' to toggle OOV words"
                  + ", or 'help' for more commands.")
            context = old_context
            continue

        interactive_output = model.predict(np.expand_dims(memory, 0),
                                           np.expand_dims(user_utt, 0),
                                           np.expand_dims(cands_mat, 0))
        pred_cand_idx = interactive_output[0]
        response = babi.candidate_answers[pred_cand_idx]

        print(response)

        if 'api_call' in response:
            db_results = issue_api_call(
                response, db, names_to_idxs, kb_text, babi)
        else:
            db_results = []


def print_memory(context):
    if not context:
        return

    max_sent_len = max(
        map(len, map(lambda z: reduce(lambda x, y: x + ' ' + y, z), context)))

    print("-" * max_sent_len)
    for sent in context:
        print(" ".join(sent))
    print("-" * max_sent_len)


def print_human_vocab(babi):
    if babi.task + 1 < 6:
        print([x for x in babi.vocab if 'resto' not in x])
    else:
        print(babi.vocab)


def print_help():
    print(
        "Available Commands: \n"
        + " >> help: Display this help menu\n"
        + " >> exit / quit: Exit interactive mode\n"
        + " >> restart / clear: Restart the conversation and erase the bot's memory\n"
        + " >> vocab: Display usable vocabulary\n"
        + " >> allow_oov: Allow out of vocab words to be replaced with <OOV> token\n"
        + " >> show_memory: Display the current contents of the bot's memory\n")


def build_kb_db(babi):
    """
    Build a searchable database from the kb files to be used in interactive mode
    """
    with open(babi.kb_file, 'r') as f:
        kb_text = f.readlines()

    kb_text = [x.replace('\t', ' ') for x in kb_text]

    db = {}

    property_types = set(x.split(' ')[2] for x in kb_text)

    for ptype in property_types:
        unique_props = set(x.split(' ')[3].strip()
                           for x in kb_text if ptype in x.strip().split(' '))

        db[ptype] = {prop: [x for idx, x in enumerate(
            kb_text) if prop in x.strip().split(' ')] for prop in unique_props}
        db[ptype][ptype] = kb_text

    resto_names = set(x.split(' ')[1] for x in kb_text)
    names_to_idxs = {}
    for name in resto_names:
        names_to_idxs[name] = [idx for idx, x in enumerate(
            kb_text) if name in x.strip().split(' ')]

    kb_text_clean = np.array(
        [' '.join(x.strip().split(' ')[1:]) for x in kb_text])

    return db, names_to_idxs, kb_text_clean


def issue_api_call(api_call, db, names_to_idxs, kb_text, babi):
    """
    Parse the api call and use it to search the database
    """
    desired_properties = api_call.strip().split(' ')[1:]

    if babi.task + 1 < 6:
        properties_order = ['R_cuisine', 'R_location', 'R_number', 'R_price']
    else:
        properties_order = ['R_cuisine', 'R_location', 'R_price']

    assert len(properties_order) == len(desired_properties)

    # Start result as all possible kb entries
    returned_kb_idxs = set(
        itertools.chain.from_iterable(
            names_to_idxs.values()))

    for ptype, prop in zip(properties_order, desired_properties):
        kb_idxs = [names_to_idxs[x.split(' ')[1]] for x in db[ptype][prop]]
        kb_idxs = list(itertools.chain.from_iterable(kb_idxs))
        # iteratively perform intersection with subset that matches query
        returned_kb_idxs = returned_kb_idxs.intersection(kb_idxs)

    returned_kb_idxs = list(returned_kb_idxs)
    # Return actual text kb entries
    kb_results = list(kb_text[returned_kb_idxs])
    return kb_results
