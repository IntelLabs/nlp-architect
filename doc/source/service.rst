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

Using as a REST service
#######################


Overview
========
NLP Architect server is a Falcon server for serving predictions to different models in NLP Architect repository.
The server also includes a very very simple front-end web exposing the model's annotations using displaCy and displaCyENT visualizations.
The server supports both `json` and `gzip` formats.

Running NLP Architect server
============================
In order to keep NLP Architect server light and simple as possible, the server uploads only the needed model.
Hence, in order to run the server you need to know which service you want to upload.
Currently we provided 2 services:

 1. `bist` service which provides BIST Dependency parsing
 2. `spacy_ner` service which provides Spacy NER annotations.

In order to run the server, simply run `serve.py` with the Parameter `--name` as the name of the service you wish to serve.
Once the model is loaded, the server will run on `http://localhost:8080/{service_name}`.

If you wish to use the server's visualization - enter `http://localhost:8080/{service_name}/demo.html`

Otherwise the expected Request for the server is the following:

.. code:: json

    {"docs":
      [
        {"id": 1,
         "doc": "Time flies like an arrow. fruit flies like a banana."},
        {"id": 2,
         "doc": "the horse passed the barn fell"},
        {"id": 3,
         "doc": "the old man the boat"}
       ]
     }

Request Headers
---------------

- Content-Type: "application/json" or "application/gzip"

- Response-Format: The response format, "json" or "gzip". The default response format is json.

The server supports 2 types of Responses (see `Annotation Structure Types - Server Responses` bellow).

Examples for running NLP Architect server
=========================================
We currently support only 2 services:

- BIST parser - Core NLP models annotation structure

.. code:: python

    python server/serve.py --name bist

Once the server is up and running you can go to `http://localhost:8080/bist/demo.html`
and check out a few test sentences, or you can send a POST request (as described above)
to `http://localhost:8080/bist`, and receive `CoreNLPDoc` annotation structure response.

.. image :: assets/bist_service.png

- Spacy NER - High-level models annotation structure

.. code:: python

    python server/serve.py --name spacy_ner

Once the server is up and running you can go to `http://localhost:8080/spacy_ner/demo.html`
and check out a few test sentences, or you can send a Post request (as described above)
to `http://localhost:8080/spacy_ner`, and receive `HighLevelDoc` annotation structure response.

.. image :: assets/spacy_ner_service.png

You can also take a look at the tests (tests/nlp_architect_server) to see more examples.

Example CURL request
--------------------

Running `spacy_ner` model

.. code:: json

    curl -i -H "Response-Format:json" -H "Content-Type:application/json" -d '{"docs": [{"id": 1,"doc": "Intel Corporation is an American multinational corporation and technology company headquartered in Santa Clara, California, in the Silicon Valley."}]}' http://{localhost_ip}:8080/spacy_ner


Running `bist` model

.. code:: json

    curl -i -H "Response-Format:json" -H "Content-Type:application/json" -d '{"docs":[{"id": 1,"doc": "Time flies like an arrow. fruit flies like a banana."},{"id": 2,"doc": "the horse passed the barn fell"},{"id": 3,"doc": "the old man the boat"}]}' http://10.13.133.120:8080/bist


Annotation Structure Types - Server Responses
=============================================
The server supports 2 types of annotation structure (responses from the server):

-  **Core NLP models annotation structure**:

A annotation of a Core NLP model (POS, LEMMA, dependency relations etc.). usually a word-to-label annotation used for the lower level of NLP task.

-  **High-level models annotation structure**:

An annotation of a more high-level model (Intent Extraction, NER, Noun-Phrase chunking, etc.). usually a span-to-label annotation used for higher
level of nlp tasks and applications.

Core NLP models annotation structure
------------------------------------
`CoreNLPDoc` class is hosting the Core NLP models annotation structure.
(can be imported using: `from nlp_architect.utils.core_nlp_models_doc import CoreNLPDoc`).

.. code:: json

    {
        "doc_text" : "<the_document_text>" (string)
        "sentences" : list of sentences, each word in a sentence is represented in a dict (list(list(dict))). the dict is structured as follows:
                    {
                        "start": <start_index> (int),
                        "len": <word_length> (int),
                        "pos": <POS_label> (string),
                        "ner": <NER_label> (string),
                        "lemma": <Lemma_string> (string),
                        "gov": <GOV_index> (int),
                        "rel": <Dependency_Relation_label> (string)
                     }
    }


High-level models annotation structure
--------------------------------------
`HighLevelDoc` class is hosting the High-level models annotation structure.
(can be imported using: `from nlp_architect.utils.high_level_models_doc import HighLevelDoc`).

.. code:: json

    {
        "doc_text" : "<the_document_text>" (string)
        "annotation_set" : list of all annotations in document (list(string))
        "spans" : list of span dict (list(dict)), each span_dict is structured as follows:
                {
                    "end": <end_index> (int),
                    "start": <start_index> (int),
                    "type": <annotation_string> (string)
                 }

NLP Architect server - developers guide
=======================================
This section is for developers who wish to add a new service to NLP-Architect server.

Adding a new service to the server
----------------------------------
All the services are documented in `services.json` file under `nlp_architect_server` folder (each key is a service name).

In order to add a new service to the server you need to go over 3 steps:

1. Choose the type of your service: Core NLP models or High-level models

2. Create API for your service. Create the file under `nlp_architect/api/abstract_api` folder. Make sure your class inherits from `AbstractApi` (`from nlp_architect.api.abstract_api import AbstractApi`) and implements all its methods. Notice that your `inference` class_method must return either "CoreNLPDoc" or "HighLevelDoc".

3. Add new service to `services.json` in the following template:

.. code:: json

    "<service_name>" : {"file_name": "<api_file_name>", "type": "core"\"high_level"}
