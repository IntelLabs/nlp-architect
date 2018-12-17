.. ---------------------------------------------------------------------------
.. Copyright 2017-2018 Intel Corporation
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

End-to-End Memory Networks for Goal Oriented Dialogue
#####################################################

Overview
========
This directory contains an implementation of an End-to-End Memory Network for goal oriented dialogue in TensorFlow.

Goal oriented dialogue is a subset of open-domain dialogue where an automated agent has a specific
goal for the outcome of the interaction. At a high level, the system needs to understand a user
request and complete a related task with a clear goal within a limited number of dialog turns.
This task could be making a restaurant reservation, placing an order, setting a timer, or many of the digital personal assistant tasks.

End-to-End Memory Networks are generic semi-recurrent neural networks which allow for a bank of
external memories to be read from and used during execution. They can be used in place of traditional
slot-filling algorithms to accomplish goal oriented dialogue tasks without the need for expensive
hand-labeled dialogue data. End-to-End Memory Networks have also been shown to be useful for
Question-Answering and information retrieval tasks.

**End-to-End Memory Network**

.. image:: https://camo.githubusercontent.com/ba1c7dbbccc5dd51d4a76cc6ef849bca65a9bf4d/687474703a2f2f692e696d6775722e636f6d2f6e7638394a4c632e706e67
    :alt: n2n_memory_networks

**Goal Oriented Dialog**

.. image:: https://i.imgur.com/5pQJqjM.png
    :alt: goal_oriented_dialog


Files
=====
- **nlp_architect/data/babi_dialog.py**: Data loader :py:class:`class <nlp_architect.data.babi_dialog.BABI_Dialog>` to download data if not present and perform preprocessing.
- **nlp_architect/models/memn2n_dialogue.py**: Implementation of :py:class:`MemN2N_Dialog <nlp_architect.models.memn2n_dialogue.MemN2N_Dialog>` class for dialogue tasks.
- **examples/memn2n_dialog/train_model.py**: Training script to load dataset and train memory network.
- **examples/memn2n_dialog/interactive.py**: Inference script to run interactive session with a trained goal oriented dialog agent.
- **examples/memn2n_dialog/interactive_utils.py**: Utilities to support interactive mode and simulate backend database.

Datasets
========
The dataset used for training and evaluation is under the umbrella of the Facebook bAbI dialog tasks
(https://research.fb.com/downloads/babi/, License: https://github.com/facebook/bAbI-tasks/blob/master/LICENSE.md). The terms and conditions of the data set license apply. Intel does not grant any rights to the data files. The dataset is automatically downloaded if not found,
and the preprocessing all happens at the beginning of training.

There are six separate tasks, tasks 1 through 5 are from simulated conversations between a customer
and a restaurant booking bot (created by Facebook), and task 6 is more realistic natural language
restaurant booking conversations as part of the `dialog state tracking challenge`_.

The descriptions of the six tasks are as follow:

- bAbI dialog dataset:
    - Task 1: Issuing API Calls
    - Task 2: Updating API Calls
    - Task 3: Displaying Options
    - Task 4: Providing Extra Information
    - Task 5: Conducting Full Dialogs

- Dialog State Tracking Challenge 2 Dataset:
    - Task 6: DSTC2 Full Dialogs

Running Modalities
==================

Training
--------
To train the model without match type on full dialog tasks, the following command can be used:

.. code:: python

  python examples/memn2n_dialog/train_model.py --task 5 --weights_save_path memn2n_weights.npz

The flag ``--use_match_type`` can also be used to enable match type features (for improved out-of-vocab performance but slower training).

Interactive Mode
----------------
To begin interactive evaluation with a trained model, the following command can be used:

.. code:: python

  python examples/memn2n_dialog/interactive.py --model_file memn2n_weights.npz

Interactive evaluation begins at the end of training and works as an interactive shell.
Commands available for the shell are as follows:

- help: Display this help menu
- exit / quit: Exit interactive mode
- restart / clear: Restart the conversation and erase the bot's memory
- vocab: Display usable vocabulary
- allow_oov: Allow out of vocab words to be replaced with <OOV> token
- show_memory: Display the current contents of the bot's memory
- show_attention: Show the bot's memory & associated computed attention for the last memory hop

Otherwise, the interactive mode operates as a chat bot, responding to dialog to assist with
restaurant booking. Vocabulary of the model is limited, please use the vocab command to see what the
model actually understands.

Results
=======
The model was trained and evaluated on the 6 bAbI Dialog tasks with the following results.

.. csv-table::
  :header: "Task", "This", "Published", "This (w/ match-type)", "Published (w/ match-type)"
  :widths: 20, 20, 20, 20, 20
  :escape: ~

  1, 99.8, 99.9, 100.0, 100.0
  2, 100.0, 100.0, 100.0, 98.3
  3, 74.8, 74.9, 74.6, 74.9
  4, 57.2, 59.5, 100.0, 100.0
  5, 96.4, 96.1, 95.6, 93.4
  6, 48.1, 41.1, 45.4, 41.0

References
==========
- **Paper**: A. Bordes, Y. Boureau, J. Weston. `Learning End-to-End Goal-Oriented Dialog`_ 2016
- **Reference TF Implementation**: `chatbot-MemN2N-tensorflow`_ (no match-type or interactive mode)

.. _Learning End-to-End Goal-Oriented Dialog: https://arxiv.org/abs/1605.07683
.. _chatbot-MemN2N-tensorflow: https://github.com/vyraun/chatbot-MemN2N-tensorflow
.. _dialog state tracking challenge: https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/
