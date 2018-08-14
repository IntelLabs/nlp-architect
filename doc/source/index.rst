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

NLP Architect by Intel® AI Lab
###############################

| **Release:**  |version|
| **Date:**     |today|

"""""""""""""""""""""""""""""""

NLP Architect is an open-source Python library for exploring the state-of-the-art deep learning topologies and techniques for natural language processing and natural
language understanding. It is intended to be a platform for future research and
collaboration.

The library includes our past and ongoing NLP research and development efforts as part of Intel AI Lab.

NLP Architect can be downloaded from Github: https://github.com/NervanaSystems/nlp-architect

How can NLP Architect be used
=============================

- Train models using provided algorithms, reference datasets and configurations
- Train models using your own data
- Create new/extend models based on existing models or topologies
- Explore how deep learning models tackle various NLP tasks
- Experiment and optimize state-of-the-art deep learning algorithms
- integrate modules and utilities from the library to solutions


Library Overview
================

Research driven NLP/NLU models
``````````````````````````````
The library contains state-of-art and novel NLP and NLU models in a variety of topics:

- Dependency parsing
- Intent detection and Slot tagging model for Intent based applications
- Memory Networks for goal-oriented dialog
- Key-value Network for question&answer system
- Noun phrase embedding vectors model
- Noun phrase semantic segmentation
- Named Entity Recognition
- Word Chunking
- Reading comprehension
- Language modeling using Temporal Convolution Network
- Unsupervised Crosslingual Word Embedding
- Supervised sentiment analysis


Deep Learning frameworks
````````````````````````
Because of the current research nature of the library, several open source deep learning frameworks are used in this repository including:

- `Intel® Nervana™ graph`_
- `Intel® neon`_
- Tensorflow_ or `Intel-Optimized TensorFlow`_
- Dynet_
- Keras_

Overtime the list of models and frameworks included in this space will change, though all generally run with Python 3.5+

.. note::

  The default installations script use CPU-based installations of Tensorflow/Dynet/Neon/nGraph. To install GPU supported binaries please refer to the framework's website for installation instructions.


Using the Models
````````````````
Each of the models includes a comprehensive description on algorithms, network topologies, reference dataset descriptions and loader, and evaluation results. Overtime the list of models included in this space will grow.


Contributing to the library
````````````````````````````
We welcome collaboration, suggestions, and critiques. For information on how to become a developer
on this project, please see the :doc:`developer guide <developer_guide>`.


.. _Intel® neon: https://github.com/nervanasystems/neon
.. _Intel® Nervana™ graph: https://github.com/NervanaSystems/ngraph-python
.. _Tensorflow: https://www.tensorflow.org/
.. _Intel-Optimized TensorFlow: https://software.intel.com/en-us/articles/intel-optimized-tensorflow-wheel-now-available
.. _Keras: https://keras.io/
.. _Dynet: https://dynet.readthedocs.io/en/latest/


.. toctree::
   :hidden:
   :maxdepth: 1

   overview.rst
   installation.rst
   publications.rst
   tutorials.rst
   REST Server <service.rst>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: NLP/NLU Components

   chunker.rst
   ner_crf.rst
   intent.rst
   np_segmentation.rst
   bist_parser.rst
   word_sense.rst
   np2vec.rst
   supervised_sentiment.rst
   TCN Language Model <tcn.rst>
   Unsupervised Crosslingual Embeddings <crosslingual_emb.rst>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: End to End Models

   reading_comprehension.rst
   memn2n.rst
   kvmemn2n.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Solutions

   Set Expansion <term_set_expansion.rst>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Pipelines

   spacy_bist.rst
   spacy_np_annotator.rst

.. toctree::
  :hidden:
  :maxdepth: 1
  :caption: For Developers

  developer_guide.rst
  api.rst

.. _https://github.com/NervanaSystems/nlp-architect: https://github.com/NervanaSystems/nlp-architect
