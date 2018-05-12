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

Core Library Modules
#####################

Overview
========

NLP Architect core consists of a set of NLP libraries that can be used
as building blocks for language understanding. These libraries are built with
modularity in mind, so that they can be utilized much like you would used
spacy_ or nltk_.

When completed, these libraries will be part of the NLP Services Server as imaged below.

.. image :: assets/core_library.png

The current core modules are:

* :doc:`NP2Vec`<np2vec>`: word embedding's model training for Noun Phrases
* :doc:`Chunker <chunker>`: syntactical part of speech tagging
* :doc:`Intent <intent>`: extraction of the intended action from text
* :doc:`Noun-phrase Segmentation <np_segmentation>`: parsing nouns and phrases
* :doc:`NER Expansion <ne_expansion>`: a domain agnostic semi-supervised algorithm for named entity set expansion
* :doc:`Most Common Word Sense <word_sense>`: determination of the most probable meaning of a word

Each of these modules are described in depth in the next sections


.. _spacy: https://spacy.io/
.. _nltk: http://www.nltk.org/
