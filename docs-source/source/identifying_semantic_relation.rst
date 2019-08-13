.. _identifying_semantic_relation:

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

Identifying Semantic Relations
##############################

Overview
========
Semantic relation identification is the task of determining whether there is a relation between two entities. Those entities could be event mentions (referring to verbs and actions phrases) or entity mentions (referring to objects, locations, persons, time, etc.).
Described below are 6 different methods for extraction relations using external data resources: Wikipedia, Wordnet, Word embeddings, Computational, Referent-Dictionary and VerbOcean.

Each semantic relation identifier below is capable of identifying a set of pre-defined relation types between two events or two entity mentions.

.. note::

    Each relation identifier extractor can be configured to initialize and run in different modes as described below in the *Initialization options* code example sections, this refers to working online directly against the dataset website, a locally stored resource dataset, or a snapshot of the resource containing only relevant data (created according to some input dataset defined by the user).

    In order to prepare a resource snapshot refer to `Downloading and generating external resources Data`_.

Wikipedia
---------

* Use :py:class:`WikipediaRelationExtraction <nlp_architect.data.cdc_resources.relations.wikipedia_relation_extraction.WikipediaRelationExtraction>` model to extract relations based on Wikipedia page information.

* Supports: Event and Entity mentions.

Relation types
~~~~~~~~~~~~~~

* Redirect Links: the two mentions have the same Wikipedia redirect link (see: `Wiki-Redirect <https://en.wikipedia.org/wiki/Wikipedia:Redirect>`_ for more details)
* Aliases: one mention is a Wikipedia alias of the other input mention (see: `Wiki-Aliases <https://www.wikidata.org/wiki/Help:Aliases>`_ for more details)
* Disambiguation: one input mention is a Wikipedia disambiguation of the other input mention (see: `Wiki-Disambiguation <https://en.wikipedia.org/wiki/Category:Disambiguation_pages>`_ for more details)
* Category: one input mention is a Wikipedia category of the other input mention (see: `Wiki-Category <https://en.wikipedia.org/wiki/Help:Category>`_ for more details)
* Title Parenthesis: one input mention is a Wikipedia title parenthesis of the other input mention (see: `Extracting Lexical Reference Rules from Wikipedia <http://u.cs.biu.ac.il/~dagan/publications/ACL09%20camera%20ready.pdf>`_ for more details)
* Be-Comp / Is-A: one input mention has a 'is-a' relation which contains the other input mention (see: `Extracting Lexical Reference Rules from Wikipedia <http://u.cs.biu.ac.il/~dagan/publications/ACL09%20camera%20ready.pdf>`_ for more details)

Initialization options
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # 3 methods for Wikipedia extractor initialization (running against wiki web site, data sub-set or local elastic DB)
    # Online initialization for full data access against Wikipedia site
    wiki_online = WikipediaRelationExtraction(WikipediaSearchMethod.ONLINE)
    # Or use offline initialization if created a snapshot
    wiki_offline = WikipediaRelationExtraction(WikipediaSearchMethod.OFFLINE, ROOT_DIR + '/mini_wiki.json')
    # Or use elastic initialization if you created a local database of wikipedia
    wiki_elastic = WikipediaRelationExtraction(WikipediaSearchMethod.ELASTIC, host='localhost', port=9200, index='enwiki_v2')

Wordnet
-------

* Use :py:class:`WordnetRelationExtraction <nlp_architect.data.cdc_resources.relations.wordnet_relation_extraction.WordnetRelationExtraction>` to extract relations based on WordNet.

* Support: Event and Entity mentions.

Relation types
~~~~~~~~~~~~~~

* Derivationally - Terms in different syntactic categories that have the same root form and are semantically related
* Synset - A synonym set; a set of words that are interchangeable in some context without changing the truth value of the preposition in which they are embedded

See: `WordNet Glossary <https://wordnet.princeton.edu/documentation/wngloss7wn>`_ for more details.


Initialization options
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # 2 methods for Wordnet extractor initialization (Running on original data or on a sub-set)
    # Initialization for full data access
    wn_online = WordnetRelationExtraction(OnlineOROfflineMethod.ONLINE)
    # Or use offline initialization if created a snapshot
    wn_offline = WordnetRelationExtraction(OnlineOROfflineMethod.OFFLINE, wn_file=ROOT_DIR + '/mini_wn.json')

Verb-Ocean
----------

* Use :py:class:`VerboceanRelationExtraction <nlp_architect.data.cdc_resources.relations.verbocean_relation_extraction.VerboceanRelationExtraction>` to extract relations based on `Verb-Ocean <http://demo.patrickpantel.com/demos/verbocean/>`_.

* Support: Event mentions only.

Initialization options
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # 2 method for VerbOcean extractor initialization (with original data or a sub-set)
    # Initialization for full data access
    vo_online = VerboceanRelationExtraction(OnlineOROfflineMethod.ONLINE, ROOT_DIR + '/verbocean.unrefined.2004-05-20.txt')
    # Or use offline initialization if created a snapshot
    vo_offline = VerboceanRelationExtraction(OnlineOROfflineMethod.OFFLINE, ROOT_DIR + '/mini_vo.json')

© Timothy Chklovski and Patrick Pantel 2004-2016; All Rights Reserved. With any questions, contact Timothy Chklovski or Patrick Pantel.

Referent-Dictionary
-------------------

* Use :py:class:`ReferentDictRelationExtraction <nlp_architect.data.cdc_resources.relations.referent_dict_relation_extraction.ReferentDictRelationExtraction>` to extract relations based on `Referent-Dict <http://www.aclweb.org/anthology/N13-1110>`_.

* Support: Entity mentions only.

Initialization options
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # 2 methods for ReferentDict extractor initialization (with original data or a sub-set)
    # Initialization for full data access
    ref_dict_onine = ReferentDictRelationExtraction(OnlineOROfflineMethod.ONLINE, ROOT_DIR   '/ref.dict1.tsv')
    # Or use offline initialization if created a snapshot
    ref_dict_offline = ReferentDictRelationExtraction(OnlineOROfflineMethod.OFFLINE, ROOT_DIR + '/mini_dict.json')

© Marta Recasens, Matthew Can, and Dan Jurafsky. 2013. Same Referent,
Different Words: Unsupervised Mining of Opaque Coreferent
Mentions. Proceedings of NAACL 2013.

Word Embedding
--------------

* Use :py:class:`WordEmbeddingRelationExtraction <nlp_architect.data.cdc_resources.relations.word_embedding_relation_extraction.WordEmbeddingRelationExtraction>` to extract relations based on w2v distance.

* Support: Event and Entity mentions.

Supported Embeddings types
~~~~~~~~~~~~~~~~~~~~~~~~~~

* `Elmo <https://allennlp.org/elmo>`_ - For using pre-trained Elmo embeddings
* `Glove <https://nlp.stanford.edu/projects/glove>`_ - Using pre-trained Glove embeddings

Initialization options
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # 4 flavors of Embedding model initialization (running Elmo, Glove or data sub-set of them)
    # Initialization for Elmo Pre-Trained vectors
    embed_elmo_online = WordEmbaddingRelationExtraction(EmbeddingMethod.ELMO)
    embed_elmo_offline = WordEmbaddingRelationExtraction(EmbeddingMethod.ELMO_OFFLINE, glove_file='ROOT_DIR + '/elmo_snippet.pickle')
    # Embedding extractor initialization (GloVe)
    # Initialization of Glove Pre-Trained vectors
    embed_glove_online = WordEmbeddingRelationExtraction(EmbeddingMethod.GLOVE, glove_file='ROOT_DIR + '/glove.840B.300d.txt')
    # Or use offline initialization if created a snapshot
    embed_glove_offline = WordEmbaddingRelationExtraction(EmbeddingMethod.GLOVE_OFFLINE, glove_file='ROOT_DIR + '/glove_mini.pickle')

Computational
-------------

* Use :py:class:`ComputedRelationExtraction <nlp_architect.data.cdc_resources.relations.computed_relation_extraction.ComputedRelationExtraction>` to extract relations based on rules such as Head match and Fuzzy Fit.

* Support: Event and Entity mentions.

Relation types
~~~~~~~~~~~~~~

* Exact Match: Mentions are identical
* Fuzzy Match: Mentions are fuzzy similar
* Fuzzy Head: Mentions heads are fuzzy similar (in cases mentions are more then a single token)
* Head Lemma: Mentions have the same head lemma (in cases mentions are more then a single token)

Initialization
~~~~~~~~~~~~~~

.. code:: python

    # 1 method fpr Computed extractor initialization
    computed = ComputedRelationExtraction()

Examples
--------

* Using Wikipedia Relation identifier for mentions of *'IBM'* and *'International Business Machines'* will result with the following relation types: ```WIKIPEDIA_CATEGORY, WIKIPEDIA_ALIASES, WIKIPEDIA_REDIRECT_LINK```

* Using WordNet Relation identifier for mentions of *'lawyer'* and *'attorney'* will result with the following relations types: ```WORDNET_SAME_SYNSET, WORDNET_DERIVATIONALLY```

* Using Referent-Dict Relation identifier for mentions of *'company'* and *'apple'* will result with ```REFERENT_DICT``` relation type.

* Using VerbOcean Relation identifier for mentions of *'expedite'* and *'accelerate'* will result with ```VERBOCEAN_MATCH``` relation type.

Code Example
~~~~~~~~~~~~

Each relation identifier implements two main methods to identify the relations types:

1) ``extract_all_relations()`` - Extract all supported relations types from this relation model
2) ``extract_sub_relations()`` - Extract particular relation type, from this relation model

See detailed example below and methods documentation for more details on how to use the identifiers.

.. code:: python

    computed = ComputedRelationExtraction()
    ref_dict = ReferentDictRelationExtraction(OnlineOROfflineMethod.ONLINE,
                                              '<replace with Ref-Dict data location>')
    vo = VerboceanRelationExtraction(OnlineOROfflineMethod.ONLINE,
                                     '<replace with VerbOcean data location>')
    wiki = WikipediaRelationExtraction(WikipediaSearchMethod.ONLINE)
    embed = WordEmbaddingRelationExtraction(EmbeddingMethod.ELMO)
    wn = WordnetRelationExtraction(OnlineOROfflineMethod.ONLINE)

    mention_x1 = MentionDataLight(
        'IBM',
        mention_context='IBM manufactures and markets computer hardware, middleware and software')
    mention_y1 = MentionDataLight(
        'International Business Machines',
        mention_context='International Business Machines Corporation is an '
                        'American multinational information technology company')

    computed_relations = computed.extract_all_relations(mention_x1, mention_y1)
    ref_dict_relations = ref_dict.extract_all_relations(mention_x1, mention_y1)
    vo_relations = vo.extract_all_relations(mention_x1, mention_y1)
    wiki_relations = wiki.extract_all_relations(mention_x1, mention_y1)
    embed_relations = embed.extract_all_relations(mention_x1, mention_y1)
    wn_relaions = wn.extract_all_relations(mention_x1, mention_y1)

You can find the above example in this location: ``examples/cross_doc_coref/relation_extraction_example.py``

Downloading and generating external resources data
==================================================
This section describes how to download resources required for relation identifiers and how to prepare resources for working locally or with a snapshot of a resource.

Full External Resources
-----------------------

* `Referent-Dict <http://nlp.stanford.edu/pubs/coref-dictionary.zip>`_, used in ``ReferentDictRelationExtraction``
* `Verb-Ocean <http://www.patrickpantel.com/cgi-bin/web/tools/getfile.pl?type=data&id=verbocean/verbocean-verbs.2004-05-20.txt>`_ used in ``VerboceanRelationExtraction``
* `Glove <https://nlp.stanford.edu/projects/glove/>`_ used in ``WordEmbeddingRelationExtraction``

Generating resource snapshots
-----------------------------
Using a large dataset with relation identifiers that work by querying an online resource might take a lot of time due to network latency and overhead. In addition, capturing an online dataset is useful for many train/test tasks that the user might do. For this purpose we included scripts to capture a snapshot (or a subset) of an online resource.
The downloaded snapshot can be loaded using the relation identifiers as data input.

Each script requires a **mentions** file in JSON format as seen below. This file must contain the event or entity mentions that the user is interested it (or the subset of data needed to be captured):

.. code-block:: JSON

    [
        { # Mention 1
            "tokens_str": "Intel" #Required,
            "context": "Intel is the world's second largest and second highest valued semiconductor chip maker" #Optional (used in Elmo)
        },
        { # Mention 2
            "tokens_str": "Tara Reid"
        },
        ...
    ]


Generate Scripts
~~~~~~~~~~~~~~~~

**Generate ReferentDict:**

::

    python -m nlp_architect.data.cdc_resources.gen_scripts.create_reference_dict_dump --ref_dict=<ref.dict1.tsv downloaded file> --mentions=<in_mentions.json> --output=<output.json>

**Generate VerbOcean:**

::

    python -m nlp_architect.data.cdc_resources.gen_scripts.create_verbocean_dump --vo=<verbocean.unrefined.2004-05-20.txt downloaded file> --mentions=<in_mentions.json> --output=<output.json>

**Generate WordEmbedding Glove:**

::

    python -m nlp_architect.data.cdc_resources.gen_scripts.create_word_embed_glove_dump --mentions=<in_mentions.json> --glove=glove.840B.300d.txt --output=<output.pickle>

**Generate Wordnet:**

::

    python -m nlp_architect.data.cdc_resources.gen_scripts.create_wordnet_dump --mentions=<in_mentions.json> --output=<output.json>

**Generate Wikipedia:**

::

    python -m nlp_architect.data.cdc_resources.gen_scripts.create_wiki_dump --mentions=<in_mentions.json> --output=<output.json>``

.. note::

     **For a fast evaluation using Wikipedia at run time**, on live data, there is an option to generate a local ElasticSearch database of the entire Wiki site using this resource: `Wiki to Elastic <https://github.com/AlonEirew/wikipedia-to-elastic/>`_, It is highly recommended since using online evaluation against Wikipedia site can be very slow.
    In case you adopt elastic local database, Initiate ``WikipediaRelationExtraction`` relation extraction using ``WikipediaSearchMethod.ELASTIC``

**Generate Wikipedia Snapshot using Elastic data instead of from online wikipedia site:**

::

     python -m nlp_architect.data.cdc_resources.gen_scripts.create_wiki_dump --mentions=<in_mentions.json> --host=<elastic_host eg:localhost> --port=<elastic_port eg:9200> --index=<elastic_index> --output=<output.json>``
