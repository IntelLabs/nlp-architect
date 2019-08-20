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

Cross Document Co-Reference
###########################

Overview
========

Cross Document Coreference resolution is the task of determining which event or entity mentions expressed in language refer to a similar real-world event or entity across different documents in the same topic.

Definitions:

* **Event mention** refers to verb and action phrases in a document text.
* **Entity mentions** refers to object, location, person, time and so on phrases in a document text.
* **Document** refers to a text article (with one or more sentences) on a single subject and which contains entity and event mentions.
* **Topic** refers to a set of documents the are on the same subject or topic.

Sieve-based System
==================
The cross document coreference system provided is a sieve-based system. A sieve is a logical layer that uses a single semantic relation identifier that extracts a certain relation type. See details descriptions of relation identifiers and types of relations in :ref:`Identifying Semantic Relation <identifying_semantic_relation>`.

The sieve-based system consists of a set of configurable sieves. Each sieve uses a computational rule based logic or an external knowledge resource in order to extract semantic relations between event or entity mentions pairs, with the purpose of clustering same or semantically similar relation mentions across multiple documents.

Refer to `Configuration`_ section below to see how-to configure a sieved-based system.

Results
=======
The sieve-based system was tested on ECB+ [1]_ corpus and evaluated using CoNLL F1 (Pradhan et al., 2014) metric.

The `ECB+ <http://www.newsreader-project.eu/results/data/the-ecb-corpus/>`_ corpus component consists of 502 documents that belong to 43 topics, annotated with mentions of events and their times, locations, human and non-human participants as well as with within- and cross-document event and entity coreference information.

The system achieved the following:

* Best in class results achieve on ECB+ Entity Cross Document Co-Reference (**69.8% F1**) using the sieves set *[Head Lemma, Exact Match, Wikipedia Redirect, Wikipedia Disambiguation and Elmo]*
* Best in class results achieve on ECB+ Event Cross Document Co-Reference (**79.0% F1**) using the sieves set *[Head Lemma, Exact Match, Wikipedia Redirect, Wikipedia Disambiguation and Fuzzy Head]*

.. [1] ECB+: Agata Cybulska and Piek Vossen. 2014. Using a sledgehammer to crack a nut? Lexical diversity and event coreference resolution.

In Proceedings of the 9th international conference on Language Resources and Evaluation (LREC2014)
ECB+ annotation is held copyright by Agata Cybulska, Piek Vossen and the VU University of Amsterdam.

Requirements
============
1. Make sure all intended relation identifier resources are available and configured properly. Refer to :ref:`Identifying Semantic Relation <identifying_semantic_relation>` to see how to use and configure the identifiers.
2. Prepare a JSON file with mentions to be used as input for the sieve-based cross document coreference system:

.. code-block:: JSON

    [
        {
            "topic_id": "2_ecb", #Required (a topic is a set of multiple documents that share the same subject)
            "doc_id": "1_10.xml", #Required (the article or document id this mention belong to)
            "sent_id": 0, #Optional (mention sentence number in document)
            "tokens_number": [ #Optional (the token number in sentence, will be required when using Within doc entities)
                13
            ],
            "tokens_str": "Josh", #Required (the mention text)
        },
        {
            "topic_id": "2_ecb", #Required
            "doc_id": "1_11.xml",
            "sent_id": 0,
            "tokens_number": [
                3
            ],
            "tokens_str": "Reid",
        },
            ...
    ]

* An example for an ECB+ entity mentions json file can be found here: ``<nlp architect root>/datasets/ecb/ecb_all_entity_mentions.json``
* An example for an ECB+ event mentions json file can be found here: ``<nlp architect root>/datasets/ecb/ecb_all_event_mentions.json``

Configuration
=============
There are two modes of operation:

    1) Entity mentions cross document coreference - for clustering entity mentions across multiple documents
    2) Event mentions cross document coreference - for clustering event mentions across multiple document


For each mode of operation there is a method for extraction defined in :py:class:`cross_doc_sieves <nlp_architect.models.cross_doc_sieves>`:
    - ``run_event_coref()`` - running event coreference resolution
    - ``run_entity_coref()`` - running entity coreference resolution

Each mode of operation requires a configuration. The configurations define which sieve should run, in what order and define constraints and thresholds

    - Use :py:class:`EntitySievesConfiguration <nlp_architect.models.cross_doc_coref.cdc_config.EventSievesConfiguration>` for configuring the needed sieves for computing events mentions
    - Use :py:class:`EntitySievesConfiguration <nlp_architect.models.cross_doc_coref.cdc_config.EntitySievesConfiguration>` for configuring the needed sieves for computing entities mentions

Configuring ``sieves_order`` enables control on the sieve configurations, ``sieves_order`` is a list of tuples (RelationType, threshold)

Use :py:class:`SievesResources <nlp_architect.models.cross_doc_coref.cdc_resource.SievesResources>` to set the correct paths to all files downloaded or created for the different types of sieves.


Sieve-based system flow
=======================
The flow of the sieve-based system is identical to both event and entity resolutions:

1) Load all mentions from input file (mentions json file).
2) Separate each mention to a *singleton* cluster (a cluster initiated with only one mention) and group the clusters by topic (so each topic has a set of clusters that belong to it) according to the input values.
3) Run the configured sieves system iteratively in the order determine in the ``sieves_order`` configuration parameter, For each sieve:

    1) Go over all clusters in a topic and try to merge 2 clusters at a time with current sieve RelationType
    2) Continue until no mergers are available using this RelationType

4) Continue to next sieve and repeat (3.1) on current state of clusters until no more sieves are left to run.
5) Return the clusters results.

See code example below for running a full cross document coreference evaluation or refer to the documentation for further details.

Code Example
============

You can find code example for running the system at: ``examples/cross_doc_coref/cross_doc_coref_sieves.py``
