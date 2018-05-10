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

Reading Comprehension
######################

Overview
========
This directory contains an implementation of the boundary model(b in the Figure) Match LSTM and
Answer Pointer network for Machine Reading Comprehension in Intel-Ngraph. The idea behind this
method is to build a question aware representation of the passage and use this representation as an
input to the pointer network which identifies the start and end indices of the answer.

**Model Architecture**

.. image: ../../models/ReadingComprehension/ngraph_implementation/MatchLSTM_Model.png


Files
======
- **nlp_architect/contrib/ngraph/match_lstm.py** -Contains different layers in ngraph required for the model
- **nlp_architect/utils/weight_initializers.py** - Contains functions to initialize the LSTM Cell in ngraph
- **examples/train.py** -Implements the end to end model along with the training commands
- **examples/utils.py**- Implements different utility functions to set up the data loader and to do the evaluation.


Datasets
========
This repository uses the SQuAD dataset. Two preprossessing steps are required prior to training:

1. Download the training and test data from https://rajpurkar.github.io/SQuAD-explorer/
2. Preprocess the data set using this command- `python squad_preprocess.py`

Training
========
Train the model using the following command

.. code:: python

  python train.py -bgpu --gpu_id 0

The command line options available are:

* ``--gpu_id`` select the gpu id train the model. Default is 0.
* ``--max_para_req`` enter the max length of the para to truncate the dataset.Default is 100. Currently the code has been tested for a maximum length of paragraph length of 100.
* ``--batch_size_squad`` enter the batch size (please note that 30 is the max batch size that will fit on a gpu with 12 gb memory). Default is 16.

Results
========
After training starts, you will see outputs like this:

.. code:: python

  Loading Embeddings
  creating training Set
  Train Set Size is 19260
  Dev set size is 2000
  compiling the graph
  generating transformer
  iteration = 1, train loss = 13.156427383422852
  F1_Score and EM_score are 0.0 0.0
  iteration = 21, train loss = 12.441322326660156
  F1_Score and EM_score are 8.333333333333332 0.0
  iteration = 41, train loss = 10.773386001586914
  F1_Score and EM_score are 6.25 6.25
  iteration = 61, train loss = 11.69123649597168
  F1_Score and EM_score are 6.25 6.25

Please note that after each epoch you will see the validation F1 and EM scores being printed out.
These numbers are a result of a much stricter evaluation and lower than the official evaluation numbers.

Considering the default setting, which has training set of 19260 examples and a development set of 2000 examples
after 15 epochs, you should expect to see a F1 and EM scores on the development set similar to this:

.. code:: python

  F1 Score ~35%
  EM Score ~25%
