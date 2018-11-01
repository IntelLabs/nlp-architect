<p align="center"><img src="doc/source/assets/nlp_architect_header.png" width="400"/></p>
<p align="center">
<a href="https://github.com/NervanaSystems/nlp-architect/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"/></a> <a href="http://nlp_architect.nervanasys.com"><img src="https://img.shields.io/readthedocs/pip/stable.svg"/></a> <a href="https://github.com/NervanaSystems/nlp-architect/blob/master/LICENSE"><img src="https://img.shields.io/badge/release-v0.3-blue.svg"/></a>
</p>

# NLP Architect by Intel® AI LAB

NLP Architect is an open-source Python library for exploring state-of-the-art
deep learning topologies and techniques for natural language processing and
natural language understanding. It is intended to be a platform for future
research and collaboration.

## Overview

The current version of NLP Architect includes these features that we found
interesting from both research perspectives and practical applications:

* NLP core models and NLU modules that provide best in class performance: Intent
  Extraction (IE), Name Entity Recognition (NER), Word Chunker, Dependency parser (BIST)
* Modules that address semantic understanding: co-locations, most
  common word sense, noun phrase embedding representation (NP2Vec), relation identification and cross document coreference.
* Components instrumental for conversational AI: ChatBot
  applications (Memory Networks for Dialog, Key-Value Memory Networks), Intent Extraction.
* End-to-end DL applications using and new topologies: Q&A, Machine
  Reading Comprehension, Language modeling using Temporal Convolution
  Networks (TCN), Unsupervised Cross-lingual embeddings, Sparse and quantized GNMT.
* Solutions using one or more models: Set Term expansion which
  uses the included word chunker as a noun phrase extractor and NP2Vec, Topics and trend analysis for analyzing temporal corpora.

<center> <img src="doc/source/assets/nlp_architect_diag.png"></center>

The library consists of core modules (topologies), data pipelines, utilities
and end-to-end model examples with training and inference scripts. We look at
these as a set of building blocks that were needed for implementing NLP use
cases based on our pragmatic research experience. Each of the models includes
algorithm descriptions and results in the [documentation].

Some of the components, with provided pre-trained models, are exposed as REST
service APIs through NLP Architect server. NLP Architect server is designed to
provide predictions across different models in NLP Architect. It also includes
a web front-end exposing the model annotations for visualizations. The server
supports extensions via a template for developers to add a new service. For
detailed documentation see this
[page](http://nlp_architect.nervanasys.com/service.html).

NLP Architect server in action
<center> <img src="doc/source/assets/service_cards.png"></center>

NLP Architect utilizes the following open source deep learning frameworks:

* [TensorFlow]
* [Intel-Optimized TensorFlow with MKL-DNN]
* [Dynet]

## Documentation
Framework documentation on NLP models, algorithms, and modules, and instructions
on how to contribute can be found at our main [documentation] site.

## Installation
### Prerequisites
To get started, clone our repository:

    git clone https://github.com/NervanaSystems/nlp-architect.git
    cd nlp-architect

Installing NLP Architect within a virtual environment to ensure a self-contained
environment is recommended.
To create and activate a new virtual environment:

    python3 -m venv .nlp_architect_env
    source .nlp_architect_env/bin/activate

Make sure `pip` and `setuptools` and `venv` are up to date before installing.

    pip3 install -U pip setuptools

### Backend

NLP Architect supports CPU, GPU and Intel Optimized Tensorflow (MKL-DNN).
Select the desired backend using a dedicated environment variable.

CPU (default):

    export NLP_ARCHITECT_BE=CPU

MKL-DNN (Linux OS only):

    export NLP_ARCHITECT_BE=MKL

GPU (Linux OS only):

    export NLP_ARCHITECT_BE=GPU


### Installation
NLP Architect is installed using `pip` and it is recommended to install in development mode.

Default:

    pip3 install .

Development mode:

    pip3 install -e .

Once installed, the `nlp_architect` command provides additional options to work with the library, issue `nlp_architect -h` to see all options.

## Packages

| Package                 	| Description                                          	|
|-------------------------	|------------------------------------------------------	|
| nlp_architect.api       	| Model server API interfaces                          	|
| nlp_architect.common    	| Common packages                                      	|
| nlp_architect.contrib   	| Framework extensions                                 	|
| nlp_architect.data      	| Datasets, data loaders and data classes              	|
| nlp_architect.models    	| NLP, NLU and End-to-End neural models                	|
| nlp_architect.pipelines 	| End-to-end NLP apps                                  	|
| nlp_architect.utils     	| Misc. I/O, metric, pre-processing and text utilities 	|
| examples                	| Example files for each model                         	|
| solutions               	| Solution applications                                	|
| server                  	| API Server and demos UI                              	|
| tutorials               	| Misc. Jupyter tutorials                              	|
<!--
Below are examples of NLP Architect server in action

BIST Parser UI
<center> <img src="doc/source/assets/bist_service.png"></center>

NER Parser UI
<center> <img src="doc/source/assets/ner_service.png"></center>

Spacy NER Parser UI
<center> <img src="doc/source/assets/spacy_ner_service.png"></center>-->


NLP Architect is an active space of research and development; Throughout future
releases new models, solutions, topologies and framework additions and changes
will be made. We aim to make sure all models run with Python 3.5+. We
encourage researchers and developers to contribute their work into the library.

## Disclaimer
The NLP Architect is released as reference code for research purposes. It is
not an official Intel product, and the level of quality and support may not be
as expected from an official product. Additional algorithms and environments are
planned to be added to the framework. Feedback and contributions from the open
source and NLP research communities are more than welcome.

## Contact
Contact the NLP Architect development team through Github issues or
email: nlp_architect@intel.com

[documentation]:http://nlp_architect.nervanasys.com
[Intel-Optimized TensorFlow with MKL-DNN]:https://software.intel.com/en-us/articles/intel-optimized-tensorflow-wheel-now-available
[TensorFlow]:https://www.tensorflow.org/
[Dynet]:https://dynet.readthedocs.io/en/latest/
