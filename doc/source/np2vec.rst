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

Noun Phrase to Vec
###################

Overview
========
Noun Phrases (NP) play a particular role in NLP algorithms.
This code consists in training a word embedding's model for Noun NP's using word2vec_ or fasttext_ algorithm.
It assumes that the NP's are already extracted and marked in the input corpus.
All the terms in the corpus are used as context in order to train the word embedding's model; however,
at the end of the training, only the word embedding's of the NP's are stored, except for the case of
fasttext training with word_ngrams=1; in this case, we store all the word embedding's,
including non-NP's in order to be able to estimate word embeddings of out-of-vocabulary NP's
(NP's that don't appear in the training corpora).

.. note::

  This code can be also used to train a word embedding's model on any marked corpus.
  For example, if you mark verbs in your corpus, you can train a verb2vec model.

NP's have to be marked in the corpus by a marking character between the words of the NP and as a suffix of the NP.
For example, if the marking character is "\_", the NP "Natural Language Processing" will be marked as "Natural_Language_Processing".

Files
======

- **nlp_architect/models/np2vec.py**: NP2vec model training, store and load code.
- **examples/train.py**: illustrates how to call NP2vec training and store code.
- **examples/inference.py**: illustrates how to call NP2vec load code.

Running Modalities
==================

Training
--------
To train the model in it's default mode, the following command can be used:

.. code:: python

  python main_train.py --corpus sample_corpus.json --np2vec_model_file sample_np2vec.model

Inference
----------------
To run inference with a saved model, the following command can be used:

.. code:: python

  python main_inference.py --np2vec_model_file sample_np2vec.model


More details about the hyperparameters at https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
for word2vec and https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText for fasttext.

.. _word2vec: https://code.google.com/archive/p/word2vec/
.. _fasttext: https://github.com/facebookresearch/fastText
