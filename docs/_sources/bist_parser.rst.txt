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

BIST Dependency Parser
#######################

Graph-based dependency parser using BiLSTM feature extractors
==============================================================

The techniques behind the parser are described in the `Simple and
Accurate Dependency Parsing Using Bidirectional LSTM Feature
Representations <https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198>`__ [1]_.

Usage
=====

To use the module, import it like so:

.. code:: python

    from nlp_architect.models.bist_parser import BISTModel

Training
========

Training the parser requires having a ``train.conllu`` file
formatted according to the CoNLL-U_ data format,
annotated with part-of-speech tags and dependencies.
The benchmark was performed on a Mac book pro with i7 processor. The parser achieves
an accuracy of 93.8 UAS on the standard Penn Treebank dataset (Universal Dependencies).


Basic Example
-------------

To train a parsing model with default parameters, type the following:

.. code:: python

    parser = BISTModel()
    parser.fit('/path/to/train.conllu')


Exhaustive Example
------------------

Optionally, the following model/training parameters can be supplied (overriding their default
values listed below):

.. code:: python

    parser = BISTModel(activation='tanh', lstm_layers=2, lstm_dims=125, pos_dims=25)
    parser.fit('/path/to/train.conllu', epochs=10)


Conducting Intermediate Evaluations
-----------------------------------

If a path to a development dataset file (annotated with POS tags and dependencies) is supplied,
intermediate model evaluations are conducted:

.. code:: python

    parser = BISTModel()
    parser.fit('/path/to/train.conllu', dev='/path/to/dev.conllu')

For each completed epoch, denoted by **n**, the following files will be created in the dataset's
directory:

- *dev_epoch_n_pred.conllu* - prediction results on dev file after **n** iterations.
- *dev_epoch_n_pred_eval.txt* - accuracy results of the above predictions.

Inference
=========

Once you have a trained :py:class:`BISTModel <nlp_architect.models.bist_parser.BISTModel>`, there are two acceptable input modes for running inference
with it. For both modes, the input must be annotated with part-of-speech tags.

File Input Mode
---------------

Supply a path to a dataset file in the CoNLL-U_ data format.

.. code:: python

    predictions = parser.predict(dataset='/path/to/test.conllu')

After running the above example, ``predictions`` will hold the input sentences with annotated
dependencies, as a collection of :py:class:`ConllEntry <nlp_architect.data.conll.ConllEntry>` objects, where each :py:class:`ConllEntry <nlp_architect.data.conll.ConllEntry>` represents an
annotated token.

ConllEntry Input Mode
---------------------

Supply a list of sentences, where each sentence is a list of annotated tokens, represented by
:py:class:`ConllEntry <nlp_architect.data.conll.ConllEntry>` instances.

.. code:: python

    predictions = parser.predict(conll='/path/to/test.conllu')

The output format is the same as in file input mode.

Evaluating Predictions
----------------------

Running an evaluation requires the following:
- Inference must be run in file input mode
- The input file must be annotated with dependencies as well

To evaluate predictions immediately after they're generated, type the following:

.. code:: python

    predictions = parser.predict(dataset='/path/to/test.conllu', evaluate=True)

This will produce 2 files in your input dataset's directory:

- *test_pred.conllu* - predictions file in CoNLL-U format
- *test_pred_eval.txt* - evaluation report text file

Saving and Loading a Model
==========================

To save a :py:class:`BISTModel <nlp_architect.models.bist_parser.BISTModel>` to some path, type:

.. code:: python

    parser.save('/path/to/bist.model')

This operation will also produce a model parameters file named *params.json*, in the same directory.
This file is required for loading the model afterwards.

To load a :py:class:`BISTModel <nlp_architect.models.bist_parser.BISTModel>` from some path, type:

.. code:: python

    parser.load('/path/to/bist.model')

Note that this operation will also look for the *params.json* in the same directory.

References
==========
.. [1] Kiperwasser, E., & Goldberg, Y. (2016). Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations. Transactions Of The Association For Computational Linguistics, 4, 313-327. https://transacl.org/ojs/index.php/tacl/article/view/885/198

.. _CoNLL-U:  http://universaldependencies.org/format.html
