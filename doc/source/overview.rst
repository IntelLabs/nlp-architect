.. ---------------------------------------------------------------------------
.. Copyright 2016-2018 Intel Corporation
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

The Intel AI NLP Architect is a repository for models exploring the state of the
art deep learning techniques for natural language processing and natural
language understanding. It is intended to be a space to promote research and
collaboration.

The library includes our past and ongoing NLP research efforts as part of Intel AI Lab.


To who is this library intended to?
===================================

- Data Scientists - as a toolkit for examining models and data to derive insights.
- Artificial Intelligence software developers - developers who want to integrate Machine Learning and Deep Learning models models as a part of a solution.
- NLP Researchers.


How would you use this library?
===============================
- Train models using provided dataset and configuration.
- Train models using their own data.
- Create new/extend models based on existing models or topologies.


Library Overview
================

Deep Learning frameworks
````````````````````````
Because of its exploratory nature, several open source deep learning frameworks are used in this repository including:

- `Intel® Nervana™ graph`_
- Intel® neon_
- Tensorflow_
- Dynet_
- Keras_

Research driven NLP models
``````````````````````````
The library contains state-of-art and novel Natural Language Processing (NLP) and Natural Language Understanding (NLU) models in a wide varity of topics:

- Dependency parsing
- Intent type detection and Slot tagging model for Intent based applications
- Memory Networks for goal-oriented dialog
- Noun phrase embedding vectors model
- Noun phrase semantic segmenatation
- Text chunking
- Reading comprehension

Using the Models
````````````````
Each of the models includes algorithm descriptions, installation
requirements, dataset descriptions and loader, and evaluation results. Overtime the list of models included in this space
will grow.
The library is compatible with Python 3.5+.

Contributing to the library
```````````````````````````
We welcome collaboration, suggestions, and critiques. For information on how to become a developer
on this project, please see the :doc:`developer guide <developer_guide>`.


.. _neon: https://github.com/nervanasystems/neon
.. _Intel® Nervana™ graph: https://github.com/NervanaSystems/ngraph-python
.. _Tensorflow: https://www.tensorflow.org/
.. _Keras: https://keras.io/
.. _Dynet: https://dynet.readthedocs.io/en/latest/
