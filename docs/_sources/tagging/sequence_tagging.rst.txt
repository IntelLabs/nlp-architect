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

==================================
Neural Models for Sequence Tagging
==================================

Overview
========

Token tagging is a core Information extraction task in which words (or phrases) are classified using a pre-defined label set. 
Common core NLP tagging tasks are Word Chunking, Part-of-speech (POS) tagging or Named entity recognition (NER).

Example
-------
Named Entity Recognition (NER) is a basic Information extraction task in which words (or phrases) are classified into pre-defined entity groups (or marked as non interesting). Entity groups share common characteristics of consisting words or phrases and are identifiable by the shape of the word or context in which they appear in sentences. Examples of entity groups are: names, numbers, locations, currency, dates, company names, etc.

Example sentence:

.. code:: bash

	John is planning a visit to London on October
	|                           |         |
	Name                        City      Date

In this example, a ``name``, ``city`` and ``date`` entities are identified.

Models
------

NLP Architect includes the following models:

* Word Chunking
* POS Tagging
* Named Entity Recognition

.. include:: chunker.rst
.. include:: ner.rst
