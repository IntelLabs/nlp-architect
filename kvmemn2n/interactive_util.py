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
from __future__ import division
from __future__ import print_function
from builtins import input
import numpy as np

from data import ex_entity_names, pad_stories, pad_values, pad_sentences


def interactive_loop(interactive_computation, wikimovies, batch_size):
    # The primary thing this interactive mode can do is for the user to ask a question
    # If none of the words in the question has a memory aspect (i.e. is not in the entities)
    # then it will respond with a "I don't know, please ask about movies..."
    while True:
        line_in = input('>>> ').strip().lower()

        if not line_in:
            line_in = "<SILENCE>"
        if line_in == 'exit' or line_in == 'quit':
            break
        if line_in == 'help':
            print_help()
            continue

        q, key = ex_entity_names(line_in, wikimovies.full_rev_entity_dict, wikimovies.re_list,
                                 return_key=True)
        if key is None or key not in wikimovies.knowledge_dict:
            print('An entity was not found in that question, please try a different question')
        else:
            related_facts = wikimovies.knowledge_dict[key]
            q = wikimovies.tokenize(q)

            data = transform_data(q, related_facts, wikimovies, batch_size)

            interactive_output = interactive_computation(data)
            pred_cand_idx = np.argmax(interactive_output['test_preds'], axis=0)[0]

            ent_word = wikimovies.index_to_word[pred_cand_idx]
            if ent_word in wikimovies.full_entity_dict:
                answer = wikimovies.full_entity_dict[ent_word]
            else:
                answer = ent_word
            print(answer)


def transform_data(q, related_facts, wikimovies, batch_size):

    k = [[wikimovies.words_to_vector(sent[0].split()) for sent in related_facts]]
    v = [[wikimovies.words_to_vector(sent[1]) for sent in related_facts]]
    q = [wikimovies.words_to_vector(q)]

    # we now need to pad it to become the correct size...
    k = np.array([pad_sentences(sents, wikimovies.story_length) for sents in k])
    k = pad_stories(k, wikimovies.story_length, wikimovies.memory_size, wikimovies.vocab_size + 1)

    v = pad_values(v, wikimovies.memory_size)
    q = pad_sentences(q, wikimovies.story_length)

    data = {'keys': k,
            'values': v,
            'query': q,
            'answer': None}

    return data


def print_help():
    print("Available Commands: \n"
          + " >> help: Display this help menu\n"
          + " >> exit / quit: Exit interactive mode\n"
          + " >> Any other text will be considered a question. \n"
          + "If you chose a question without an entity, \n"
          + "you will be promted to ask a different quesiton.")
