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

Named Entity Recognition
########################

Overview
========

Named Entity Recognition (NER) is a basic Information extraction task in which words (or phrases) are classified into pre-defined entity groups (or marked as non interesting). Entity groups share common characteristics of consisting words or phrases and are identifiable by the shape of the word or context in which they appear in sentences. Examples of entity groups are: names, numbers, locations, currency, dates, company names, etc.

Example sentence:

.. code:: bash

	John is planning a visit to London on October
	|                           |         |
	Name                        City      Date

In this example, a ``name``, ``city`` and ``date`` entities are identified.

Datasets
========

In this example we used publicly available NER datasets used in common research papers.
The data must be divided into train and test sets, preprocessed and tokenized and tagged with a finite set of entities in BIO_ format.

The dataset files must be processed into tabular format where each entry is of the following format:

.. code:: bash

	<token> <tag_1> ... <tag_n>

In the above format each sentence is separated by an empty line. Each line consists of a single sentence tokens with tags divided by white spaces (or any whitespace dividers).

Data loader
-----------

Loading data into the model can be done using the :py:class:`SequentialTaggingDataset <nlp_architect.data.sequential_tagging.SequentialTaggingDataset>` data loader which can be used with the prepared train and test data sets described above.

The data loader returns 2 Numpy matrices:
1. sparse word representation of the sentence words
2. sparse word character representation of sentence words

The user has a choice to use any representation or both when training models.

Model
=====

The NER model is based on the Bidirectional LSTM with Conditional Random Field sequence classifier published in a paper by `Lample et al.`_

The model has 2 inputs:

1. sentence words - converted into dense word embeddings or loaded from an external pre-trained word embedding model.
2. character embedding - trained using the words of the sentences.

A high level overview of the model is provided in figure below:

.. image:: assets/ner_crf_model.png

Feature generation
------------------

NER words or phrases can sometimes be easily identified by the shape of the words, by pre-built lexicons, by Part-of-speech analysis or rules combining patterns of the above features. In many other cases, those features are not known or non existent and the context in which the words appear provide the indication whether a word or a phrase is an entity.

With the help of RNN topologies we can use LSTMs to extract the character based features of words. In this model we use convolutions to extract n-grams features from the characters making up words. A similar approach with RNNs takes the last state of a BiLSTM layer as a representation of the character embeddings. More info on character embedding can be found in the paper.

Prediction layer
----------------

The main tagger model consists of a bidirectional LSTM layers. The input of the LSTM layers consists of a concatenation of the word embedding vector and the character embedding vector (provided by the character embedding network).

Finally, the output of the LSTM layers are merged into a fully-connected layer (for each token) and fed into a `Conditional Random Field classifier`_. Using CRF has been empirically shown to provide more accurate models when compared to single token prediction layers (such as a `softmax` layer).

Running Modalities
==================

Training
--------
Quick train
^^^^^^^^^^^
Train a model with default parameters given input data files:

.. code:: python

	python examples/ner/train.py --train_file train.txt --test_file test.txt

Full training parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^
All customizable parameters can be obtained by running: ``python examples/ner/train.py -h``

-h, --help            show this help message and exit
-b B                  Batch size
-e E                  Number of epochs
--train_file TRAIN_FILE
                    Train file (sequential tagging dataset format)
--test_file TEST_FILE
                    Test file (sequential tagging dataset format)
--tag_num TAG_NUM     Entity labels tab number in train/test files
--sentence_length SENTENCE_LENGTH
                    Max sentence length
--word_length WORD_LENGTH
                    Max word length in characters
--word_embedding_dims WORD_EMBEDDING_DIMS
                    Word features embedding dimension size
--character_embedding_dims CHARACTER_EMBEDDING_DIMS
                    Character features embedding dimension size
--char_features_lstm_dims CHAR_FEATURES_LSTM_DIMS
                    Character feature extractor LSTM dimension size
--entity_tagger_lstm_dims ENTITY_TAGGER_LSTM_DIMS
                    Entity tagger LSTM dimension size
--dropout DROPOUT     Dropout rate
--embedding_model EMBEDDING_MODEL
                    Path to external word embedding model file
--model_path MODEL_PATH
                    Path for saving model weights
--model_info_path MODEL_INFO_PATH
                    Path for saving model topology
--use_cudnn           use CUDNN based LSTM cells

The model will automatically save the model weights and topology information after training is complete (user can provide file names as above).

Interactive mode
----------------

The provided ``interactive.py`` file enables using a pre-trained model in interactive mode, providing input directly from stdin.

Run ``python examples/ner/interactive.py -h`` for a full list of options:

--model_path MODEL_PATH
                      Path of model weights
--model_info_path MODEL_INFO_PATH
                      Path of model topology

Quick example:

.. code:: python

	python examples/ner/interactive.py --model_path model.h5 --model_info_path model_info.dat

References
==========

1. `Neural Architectures for Named Entity Recognition`_ - Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer. 2016

.. _BIO: https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
.. _`Lample et al.`: https://arxiv.org/abs/1603.01360
.. _`Neural Architectures for Named Entity Recognition`: https://arxiv.org/abs/1603.01360
.. _`Conditional Random Field classifier`: https://en.wikipedia.org/wiki/Conditional_random_field
