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

Spacy-BIST Parser
#################


Raw text parser based on Spacy and BIST parsers
===============================================

The parser uses Spacy's english model for sentence breaking,
tokenization and token annotations (part-of-speech, lemma, NER).
Dependency relations between tokens are extracted using BIST parser.
The BIST parser is described :doc:`here <bist>`.


Dependencies
============

-  **Python** 3.5+
-  **bist** 0.1
-  **spacy** 2.0.5
-  **spacy en** 2.0.0

Usage
=====

To use the module, import it like so:

.. code:: python

    from nlp_architect.pipelines.spacy_bist.parser import SpacyBISTParser

Training
========

By default, the parser uses a pre-trained BIST model and Spacy's English
model (``spacy en``). The pre-trained BIST model is automatically
downloaded (on-demand) to ``spacy_bist/bist-pretrained/`` and then loaded
from that directory. To use other models, supply a path or link to each
model at initialization (see example below).

For instructions on how to train these models, see:
- BIST: `BIST documentation <bist.rst>`__
- Spacy: `training instructions <https://spacy.io/usage/training>`__

Example
-------

.. code:: python

    parser = SpacyBISTParser(spacy_model='/path/or/link/to/spacy/model', bist_model='/path/to/bist/model')

Parsing
=========

The parser accepts a document as a raw text string encoded in UTF-8 format and outputs a
`ParsedDocument <../utils/parsed_document.py>`__ instance which contains the annotations
(example output below).


Example
-------

.. code:: python

    parser = SpacyBISTParser()
    parsed_doc = parser.parse(doc_text='First sentence. Second sentence')
    print(parsed_doc)

Output
------

.. code:: json

    {
        "doc_text": "First sentence. Second sentence",
        "sentences": [
            [
                {
                    "start": 0,
                    "len": 5,
                    "pos": "JJ",
                    "ner": "ORDINAL",
                    "lemma": "first",
                    "gov": 1,
                    "rel": "amod",
                    "text": "First"
                },
                {
                    "start": 6,
                    "len": 8,
                    "pos": "NN",
                    "ner": "",
                    "lemma": "sentence",
                    "gov": -1,
                    "rel": "root",
                    "text": "sentence"
                },
                {
                    "start": 14,
                    "len": 1,
                    "pos": ".",
                    "ner": "",
                    "lemma": ".",
                    "gov": 1,
                    "rel": "punct",
                    "text": "."
                }
            ],
            [
                {
                    "start": 16,
                    "len": 6,
                    "pos": "JJ",
                    "ner": "ORDINAL",
                    "lemma": "second",
                    "gov": 1,
                    "rel": "amod",
                    "text": "Second"
                },
                {
                    "start": 23,
                    "len": 8,
                    "pos": "NN",
                    "ner": "",
                    "lemma": "sentence",
                    "gov": -1,
                    "rel": "root",
                    "text": "sentence"
                }
            ]
        ]
    }

Citations
=========

* Kiperwasser, E., & Goldberg, Y. (2016). Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations. Transactions Of The Association For Computational Linguistics, 4, 313-327. https://transacl.org/ojs/index.php/tacl/article/view/885/198
