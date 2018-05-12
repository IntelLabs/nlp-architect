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


Overview
########

NLP Architect is a repository for models exploring the state of the
art deep learning techniques for natural language processing and natural
language understanding. It is intended to be a space to promote research and
collaboration.

The library includes our past and ongoing NLP research efforts as part of Intel AI Lab.


To who is this library intended to?
===================================

- Data Scientists - as a toolkit for exploring models and data to derive insights
- NLP Researchers - as a set of NLP models and algorithms to use, experiment and optimize
- Machine Learning Engineers - to integrate Machine Learning and Deep Learning models models as a part of a solution


How would you use this library?
===============================
- Train models using provided algorithms, reference datasets and configurations.
- Train models using their own data.
- Create new/extend models based on existing models or topologies.


Library Overview
================

Research driven NLP/NLU models
``````````````````````````
The library contains state-of-art and novel NLP and NLU models in a varity of topics:

- Dependency parsing
- Intent detection and Slot tagging model for Intent based applications
- Memory Networks for goal-oriented dialog
- Key-value Network for question&answer system
- Noun phrase embedding vectors model
- Noun phrase semantic segmentation
- NER and NE expansion
- Text chunking
- Reading comprehension

Deep Learning frameworks
````````````````````````
Because of the current research nature of the library, several open source deep learning frameworks are used in this repository including:

- `Intel® Nervana™ graph`_
- Intel® neon_
- Tensorflow_
- Dynet_
- Keras_

Overtime the list of models included in this space will change, though all generally run with Python 3.5+


Using the Models
````````````````
Each of the models includes a comprehensive description on algorithms, network topologies, reference dataset descriptions and loader, and evaluation results. Overtime the list of models included in this space will grow.


Contributing to the library
```````````````````````````
We welcome collaboration, suggestions, and critiques. For information on how to become a developer
on this project, please see the :doc:`developer guide <developer_guide>`.


.. _neon: https://github.com/nervanasystems/neon
.. _Intel® Nervana™ graph: https://github.com/NervanaSystems/ngraph-python
.. _Tensorflow: https://www.tensorflow.org/
.. _Keras: https://keras.io/
.. _Dynet: https://dynet.readthedocs.io/en/latest/
