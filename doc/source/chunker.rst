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

Sequence Chunker
################

Overview
================

Phrase chunking is a basic NLP task that consists of tagging parts of a sentence (1 or more tokens)
syntactically, i.e. POS tagging.

.. code:: bash

	The quick brown fox jumped over the fence
	|                   |      |    |
	Noun                Verb   Prep Noun

In this example the sentence can be divided into 4 phrases, ``The quick brown fox`` and ``the fence``
are noun phrases, ``jumped`` is a verb phrase and ``over`` is a prepositional phrase.

Dataset
=======

We used the CONLL2000_ shared task dataset in our example for training a phrase chunker. More info about the CONLL2000_ shared task can be found here: https://www.clips.uantwerpen.be/conll2000/chunking/. The terms and conditions of the data set license apply. Intel does not grant any rights to the data files.


The CONLL2000_ dataset has a ``train_set`` and ``test_set`` sets consisting of 8926 and 2009 sentences annotated with Part-of-speech and chuking information.
We implemented a dataset loader, :py:class:`CONLL2000 <nlp_architect.data.sequential_tagging.CONLL2000>`, for loading and parsing :py:class:`CONLL2000 <nlp_architect.data.sequential_tagging.CONLL2000>` data into numpy arrays ready to be used sequential tagging models. For full set of options please see :py:class:`CONLL2000 <nlp_architect.data.sequential_tagging.CONLL2000>`.

The :py:class:`CONLL2000 <nlp_architect.data.sequential_tagging.CONLL2000>` loader supports the following feature generation when loading the dataset:

1. Sentence words in sparse int representation
2. Part-of-speech tags of words
3. Chunk tag of words (IOB format)
4. Characters of sentence words in sparse int representation (optional)

Model
=====

The sequence chunker is a Tensorflow-keras based model and it is implemented in :py:class:`SequenceChunker <nlp_architect.models.chunker.SequenceChunker>` and comes with several options for creating the topology depending on what input is given (tokens, external word embedding model, topology paramters).

The model is based on the paper: `Deep multi-task learning with low level tasks supervised at lower layers`_ by Søgaard and Goldberg (2016), but with minor alterations.

The described model in the paper consists of multiple sequential Bi-directional LSTM layers which are set to predict different tags. the Part-of-speech tags are projected onto a fully connected layer with softmax (in each time-stamp) after the first LSTM layer. The chunk labels are predicted similarly using a softmax layer connected to the 3rd LSTM layer.

The model's embedding vector size and LSTM layer hidden state have equal sizes, the default training optimizer is SGD with default parameters and batch size of 10.

Running Models
==============

We provide a simple example for training and running inference using the :py:class:`SequenceChunker <nlp_architect.models.chunker.SequenceChunker>` model.

``train.py`` will load CONLL2000 dataset and train a model using given training parameters (batch size, epochs, external word embedding, etc.), save the model once done training and print the performance of the model on the test set. The example supports loading GloVe/Fasttext word embedding models to be used when training a model. The training method used in this example trains on both POS and Chunk labels concurently with equal targer loss weights, this is different than what is described in the paper_.

``inference.py`` will load a saved model and a given text file with sentences and print the chunks found on the stdout.

Training
--------
Quick train
^^^^^^^^^^^
Train a model with default parameters (use sentence words and default network settings):

.. code:: python

	python train.py

Custom training parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^
All customizable parameters can be obtained by running: ``python train.py -h``

.. code:: bash

  usage: train.py [-h] [--embedding_model EMBEDDING_MODEL]
                  [--sentence_length SENTENCE_LENGTH]
                  [--feature_size FEATURE_SIZE] [--use_gpu] [-b B] [-e E]
                  [--model_name MODEL_NAME] [--print_np]

  optional arguments:
    -h, --help            show this help message and exit
    --embedding_model EMBEDDING_MODEL
                          Word embedding model path (GloVe/Fasttext/textual)
    --sentence_length SENTENCE_LENGTH
                          Maximum sentence length
    --feature_size FEATURE_SIZE
                          Feature vector size (in embedding and LSTM layers)
    --use_gpu             use GPU backend (CUDNN enabled)
    -b B                  batch size
    -e E                  number of epochs run fit model
    --model_name MODEL_NAME
                          Model name (used for saving the model)
    --print_np            Print only Noun Phrase (NP) tags accuracy

Saving the model after training is done automatically by specifying a model name with the keyword `--model_name`, the following files will be created:

* ``chunker_model.h5`` - model file
* ``chunker_model.params`` - model parameter files (topology parameters, vocabs)

Inference
---------

Running inference on a trained model using an input file (text based, each line is a document):

.. code:: python

	python inference.py --model_name <model_name> --input <input_file>.txt


.. _CONLL2000: https://www.clips.uantwerpen.be/conll2000/chunking/
.. _"https://www.clips.uantwerpen.be/conll2000/chunking/": https://www.clips.uantwerpen.be/conll2000/chunking/
.. _`Deep multi-task learning with low level tasks supervised at lower layers`: http://anthology.aclweb.org/P16-2038
.. _paper: http://anthology.aclweb.org/P16-2038
