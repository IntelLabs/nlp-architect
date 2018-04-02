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

Intel AI NLP Architect
=======================

:Release:   |version|
:Date:      |today|

The Intel AI NLP Architect is a repository for models exploring the state of the art deep learning techniques for natural language processing. It is intended to be a space to promote research and collaboration.

The library consists of 3 type of NLP/NLU components:

- Basic components
- High level models and end-to-end models
- Solutions (*TBD*)

Who is the end user of our library?

- Data Scientists - as a toolkit for examinings models and data to derive insights
- Artificial Intelligence software developers - developers who want to integrate Machine Learning and Deep Learning models models as a part of a solution
- Researchers

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Introduction

   overview.rst
   installation.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Core Modules

   core_overview.rst
   np2vec.rst
   chunker.rst
   intent.rst
   np_segmentation.rst
   ne_expansion.rst
   word_sense.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Solutions

   sentiment.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Models

   reading_comprehension.rst
   memn2n.rst
   kvmemn2n.rst


.. toctree::
  :hidden:
  :maxdepth: 1
  :caption: For Developers

  developer_guide.rst
  writing_tests.rst
