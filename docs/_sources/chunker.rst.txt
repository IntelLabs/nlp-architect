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
========

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

We used the CONLL2000_ shared task dataset in our example for training a phrase chunker. More info about the CONLL2000_ shared task can be found here: https://www.clips.uantwerpen.be/conll2000/chunking/. The terms and conditions of the data set license apply. Intel does not grant any rights to the data files. The annotation of the data has been derived from the WSJ corpus by a program written by Sabine Buchholz from Tilburg University, The Netherlands.


The CONLL2000_ dataset has a ``train_set`` and ``test_set`` sets consisting of 8926 and 2009 sentences annotated with Part-of-speech and chunking information.
We implemented a dataset loader, :py:class:`CONLL2000 <nlp_architect.data.sequential_tagging.CONLL2000>`, for loading and parsing :py:class:`CONLL2000 <nlp_architect.data.sequential_tagging.CONLL2000>` data into numpy arrays ready to be used sequential tagging models. For full set of options please see :py:class:`CONLL2000 <nlp_architect.data.sequential_tagging.CONLL2000>`.

NLP Architect has a data loader to easily load CONLL2000 which can be found in :py:class:`CONLL2000 <nlp_architect.data.sequential_tagging.CONLL2000>`. The loader supports the following feature generation when loading the dataset:

1. Sentence words in sparse int representation
2. Part-of-speech tags of words
3. Chunk tag of words (IOB format)
4. Characters of sentence words in sparse int representation (optional)


To get the dataset follow these steps:

1. download train and test files from dataset website.
2. unzip files: ``gunzip *.gz``
3. provide ``CONLL2000`` data loader or ``train.py`` sample below the directory containing the files.

Model
=====

The sequence chunker is a Tensorflow-keras based model and it is implemented in :py:class:`SequenceChunker <nlp_architect.models.chunker.SequenceChunker>` and comes with several options for creating the topology depending on what input is given (tokens, external word embedding model, topology parameters).

The model is based on the paper: `Deep multi-task learning with low level tasks supervised at lower layers`_ by Søgaard and Goldberg (2016), but with minor alterations.

The described model in the paper consists of multiple sequential Bi-directional LSTM layers which are set to predict different tags. the Part-of-speech tags are projected onto a fully connected layer and label tagging is done after the first LSTM layer. The chunk labels are predicted similarly after the 3rd LSTM layer.

The model has additional improvements to the model presented in the paper:

- Choose between Conditional Random Fields (:py:class:`CRF <nlp_architect.contrib.tensorflow.python.keras.layers.crf.CRF>`) classifier instead of 'softmax' as the prediction layers. (models using CRF have been empirically shown to produce more accurate predictions)
- Character embeddings using CNNs extracting 3-grams - extracting character information out of words was shown to help syntactic tasks such as tagging and chunking.

The model's embedding vector size and LSTM layer hidden state have equal sizes, the default training optimizer is Adam with default parameters.

Running Modalities
==================

We provide a simple example for training and running inference using the :py:class:`SequenceChunker <nlp_architect.models.chunker.SequenceChunker>` model.

``examples/chunker/train.py`` will load CONLL2000 dataset and train a model using given training parameters (batch size, epochs, external word embedding, etc.), save the model once done training and print the performance of the model on the test set. The example supports loading GloVe/Fasttext word embedding models to be used when training a model. The training method used in this example trains on both POS and Chunk labels concurrently with equal target loss weights, this is different than what is described in the paper_.

``examples/chunker/inference.py`` will load a saved model and a given text file with sentences and print the chunks found on the stdout.

Training
--------
Quick train
^^^^^^^^^^^
Train a model with default parameters (use sentence words and default network settings):

.. code:: python

	python examples/chunker/train.py --data_dir <path to CONLL2000 files>

Custom training parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^
All customizable parameters can be obtained by running: ``python train.py -h``

-h, --help            show this help message and exit
--data_dir DATA_DIR   Path to directory containing CONLL2000 files
--embedding_model EMBEDDING_MODEL
                    Word embedding model path (GloVe/Fasttext/textual)
--sentence_length SENTENCE_LENGTH
                    Maximum sentence length
--char_features       use word character features in addition to words
--max_word_length MAX_WORD_LENGTH
                    maximum number of character in one word (if
                    --char_features is enabled)
--feature_size FEATURE_SIZE
                    Feature vector size (in embedding and LSTM layers)
--use_cudnn           use CUDNN based LSTM cells
--classifier {crf,softmax}
                    classifier to use in last layer
-b B                  batch size
-e E                  number of epochs run fit model
--model_name MODEL_NAME
                    Model name (used for saving the model)

Saving the model after training is done automatically by specifying a model name with the keyword `--model_name`, the following files will be created:

* ``chunker_model.h5`` - model file
* ``chunker_model.params`` - model parameter files (topology parameters, vocabs)

Inference
---------

Running inference on a trained model using an input file (text based, each line is a document):

.. code:: python

    python examples/chunker/inference.py --model_name <model_name> --input <input_file>.txt


.. _CONLL2000: https://www.clips.uantwerpen.be/conll2000/chunking/
.. _"https://www.clips.uantwerpen.be/conll2000/chunking/": https://www.clips.uantwerpen.be/conll2000/chunking/
.. _`Deep multi-task learning with low level tasks supervised at lower layers`: http://anthology.aclweb.org/P16-2038
.. _paper: http://anthology.aclweb.org/P16-2038
