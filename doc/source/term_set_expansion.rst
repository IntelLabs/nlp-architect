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

Set Expansion Solution
######################

Overview
========
Term set expansion is the task of expanding a given partial set of terms into
a more complete set of terms that belong to the same semantic class. This
solution demonstrates the capability of a corpus-based set expansion system
in a simple web application.

.. image :: assets/expansion_demo.png

Algorithm Overview
==================
Our approach is based on representing any term of a training corpus using word embeddings in order 
to estimate the similarity between the seed terms and any candidate term. Noun phrases provide 
good approximation for candidate terms and are extracted in our system using a noun phrase chunker. 
At expansion time, given a seed of terms, the most similar terms are returned.


Training
========
   
The first step in training is to prepare the data for generating a word embedding model. We 
provide a subset of English Wikipedia at datasets/wikipedia as a sample corpus under the  
`Creative Commons Attribution-Share-Alike 3.0 License <https://creativecommons.org/licenses/by-sa/3.0/>`__ (Copyright 2018 Wikimedia Foundation).
The output of this step is the marked corpus where noun phrases are marked with the marking character (default: "\_") as described in the `NLP Architect np2vec module documentation <http://nlp_architect.nervanasys.com/np2vec.html>`__.
This is done by running:

.. code:: python

  python solutions/set_expansion/prepare_data.py --corpus TRAINING_CORPUS --marked_corpus MARKED_TRAINING_CORPUS

The next step is to train the model using `NLP Architect np2vec module <http://nlp_architect.nervanasys.com/np2vec.html>`__.
For set expansion, we recommend the following values 100, 10, 10, 0 for respectively, 
size, min_count, window and hs hyperparameters. Please refer to the np2vec module documentation for more details about these parameters.

.. code:: python

  python examples/np2vec/train.py --size 100 --min_count 10 --window 10 --hs 0 --corpus MARKED_TRAINING_CORPUS --np2vec_model_file MODEL_PATH --corpus_format txt


A `pretrained model <http://nervana-modelzoo.s3.amazonaws.com/NLP/SetExp/enwiki-20171201_pretrained_set_expansion.txt>`__
on English Wikipedia dump (enwiki-20171201-pages-articles-multistream.xml.bz2) is available under
Apache license. It has been trained with hyperparameters values
recommended above. Full English Wikipedia `raw corpus <http://nervana-modelzoo.s3.amazonaws.com/NLP/SetExp/enwiki-20171201.txt>`_ and
`marked corpus <http://nervana-modelzoo.s3.amazonaws.com/NLP/SetExp/enwiki-20171201_spacy_marked.txt>`_
are also available under the
`Creative Commons Attribution-Share-Alike 3.0 License <https://creativecommons.org/licenses/by-sa/3.0/>`__.


Inference
=========

The inference step consists of expanding given seed terms into a set of terms that belong to the same semantic class.
It can be done in two ways:

1. Running a python script:

    .. code:: python

      python solutions/set_expansion/set_expand.py --np2vec_model_file MODEL_PATH --topn TOPN

2. Web application

    A.  Loading the expand server with the trained model:

    .. code:: python

      python expand_server.py [--host HOST] [--port PORT] model_path

    The expand server gets requests containing seed terms, and expands them
    based on the given word embedding model. You can use the model you trained
    yourself in the previous step, or to provide a pre-trained model you own.
    **Important note**: default server
    will listen on localhost:1234. If you set the host/port you should also
    set it in the ui/settings.py file.


    B.  Run the UI application:

    .. code:: python

      bokeh serve --show ui

    The UI is a simple web based application for performing expansion.
    The application communicates with the server by sending expand
    requests, present the results in a simple table and export them to a csv
    file. It allows you to either directly type the terms to expand or to
    select terms from the model vocabulary list. After you get some expand
    results you can perform re-expansion by selecting terms from the results (hold the Ctrl key for
    multiple selection). **Important note**: If you set the host/port of the expand server you
    should also set it in the ui/settings.py file. You can also load the ui
    application as a server using the bokeh options --address and --port, for example:

    .. code:: python

      bokeh serve ui --address=12.13.14.15 --port=1010 --allow-websocket-origin=12.13.14.15:1010


Citation
========

`Term Set Expansion based on Multi-Context Term Embeddings: an End-to-end Workflow <https://drive.google.com/open?id=164MvUGo0-iPeuGM1b8XrH2ysZZFrzomF>`__, Jonathan Mamou,
 Oren Pereg, Moshe Wasserblat, Ido Dagan, Yoav Goldberg, Alon Eirew, Yael Green, Shira Guskin,
 Peter Izsak, Daniel Korat, COLING 2018.

