<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/NervanaSystems/nlp-architect/master/assets/nlp_architect_logo.png" width="400"/>
    <br>
<p>
<h2 align="center">
A Deep Learning NLP/NLU library by <a href="https://www.intel.ai/research/">Intel® AI Lab</a>
</h2>
<p align="center">
    <a href="https://github.com/NervanaSystems/nlp-architect/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/NervanaSystems/nlp-architect.svg?color=blue&style=popout">
    </a>
    <a href="http://nlp_architect.nervanasys.com">
        <img alt="Website" src="https://img.shields.io/website/http/nlp_architect.nervanasys.com.svg?down_color=red&down_message=offline&style=popout&up_message=online">
    </a>
    <a href="https://doi.org/10.5281/zenodo.1477518">
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.1477518.svg" alt="DOI">
    </a>
    <a href="https://pepy.tech/project/nlp-architect">
        <img src="https://pepy.tech/badge/nlp-architect"/>
    </a>
    <a href="https://github.com/NervanaSystems/nlp-architect/blob/master/LICENSE">
        <img alt="GitHub release" src="https://img.shields.io/github/release/NervanaSystems/nlp-architect.svg?style=popout">
    </a>
</p>

<h4 align="center">
  <a href="#overview">Overview</a> |
  <a href="#models">Models</a> |
  <a href="#installing-nlp-architect">Installation</a> |
  <a href="https://github.com/NervanaSystems/nlp-architect/tree/master/examples">Examples</a> <a href="http://nlp_architect.nervanasys.com/"></a> |
  <a href="http://nlp_architect.nervanasys.com">Documentation</a> |
  <a href="https://github.com/NervanaSystems/nlp-architect/tree/master/tutorials">Tutorials</a> |
  <a href="http://nlp_architect.nervanasys.com/developer_guide.html">Contributing</a>
</h4>

NLP Architect is an open source Python library for exploring state-of-the-art
deep learning topologies and techniques for Natural Language Processing and
Natural Language Understanding. NLP Architect's main purpose is to provide easy usage of NLP and NLU models while providing state-of-art and robust implementation.

## Overview

NLP Architect is an NLP library designed to be flexible, easy to extend, allow for easy and rapid integration of NLP models in applications and to showcase optimized models.

Features:

* Core NLP models used in many NLP tasks and useful in many NLP applications
* Novel NLU models showcasing novel topologies and techniques
* Simple REST API server ([doc](http://nlp_architect.nervanasys.com/service.html)):
  * serving trained models (for inference)
  * plug-in system for adding your own model
* 4 Demos of models (pre-trained by us) showcasing NLP Architect (Dependency parser, NER, Intent Extraction, Q&A)
* Based on optimized Deep Learning frameworks:
  * [TensorFlow]
  * [Intel-Optimized TensorFlow with MKL-DNN]
  * [Dynet]
* Documentation [website](http://nlp_architect.nervanasys.com/) and [tutorials](http://nlp_architect.nervanasys.com/tutorials.html)
* Essential utilities for working with NLP models - Text/String pre-processing, IO, data-manipulation, metrics, embeddings.

## Installing NLP Architect

We recommend to install NLP Architect in a new python environment, to use python 3.6+ with up-to-date `pip`, `setuptools` and `h5py`.

### Install from source (Github)

Includes core library and all content (example scripts, datasets, tutorials)

Clone repository

```sh
git clone https://github.com/NervanaSystems/nlp-architect.git
cd nlp-architect
```

Install (in develop mode)

```sh
pip install -e .
```

### Install from pypi (using `pip install`)

Includes only core library

```sh
pip install nlp-architect
```

### Further installation options

Refer to our full [installation instructions](http://nlp_architect.nervanasys.com/installation.html) page on our website for complete details on how to install NLP Architect and other backend installations such as MKL-DNN or GPU backends.

## Models

NLP models that provide best (or near) in class performance:

* [Word chunking](http://nlp_architect.nervanasys.com/chunker.html)
* [Named Entity Recognition](http://nlp_architect.nervanasys.com/ner_crf.html)
* [Dependency parsing](http://nlp_architect.nervanasys.com/bist_parser.html)
* [Intent Extraction](http://nlp_architect.nervanasys.com/intent.html)
* [Sentiment classification](http://nlp_architect.nervanasys.com/supervised_sentiment.html)
* [Language models](http://nlp_architect.nervanasys.com/tcn.html)

Natural Language Understanding (NLU) models that address semantic understanding:

* [Aspect Based Sentiment Analysis (ABSA)](http://nlp_architect.nervanasys.com/absa.html)
* [Noun phrase embedding representation (NP2Vec)](http://nlp_architect.nervanasys.com/np2vec.html)
* [Most common word sense detection](http://nlp_architect.nervanasys.com/word_sense.html)
* [Relation identification](http://nlp_architect.nervanasys.com/identifying_semantic_relation.html)
* [Cross document coreference](http://nlp_architect.nervanasys.com/cross_doc_coref.html)
* [Noun phrase semantic segmentation](http://nlp_architect.nervanasys.com/np_segmentation.html)

Components instrumental for conversational AI:

* [Joint intent detection and slot tagging](http://nlp_architect.nervanasys.com/intent.html)
* [Memory Networks for goal oriented dialog](http://nlp_architect.nervanasys.com/memn2n.html)

End-to-end Deep Learning-based NLP models:

* [Reading comprehension](http://nlp_architect.nervanasys.com/reading_comprehension.html)
* [Sparse and Quantized Neural Machine Translation (GNMT)](http://nlp_architect.nervanasys.com/sparse_gnmt.html)
* [Language Modeling using Temporal Convolution Network (TCN)](http://nlp_architect.nervanasys.com/tcn.html)
* [Unsupervised Cross-lingual embeddings](http://nlp_architect.nervanasys.com/crosslingual_emb.html)

Solutions (End-to-end applications) using one or more models:

* [Term Set expansion](http://nlp_architect.nervanasys.com/term_set_expansion.html) - uses the included word chunker as a noun phrase extractor and NP2Vec to create semantic term sets
* [Topics and trend analysis](http://nlp_architect.nervanasys.com/trend_analysis.html) - analyzing trending phrases in temporal corpora
* [Aspect Based Sentiment Analysis (ABSA)](http://nlp_architect.nervanasys.com/absa_solution.html)

## Documentation

Full library [documentation](http://nlp_architect.nervanasys.com/) of NLP models, algorithms, solutions and instructions
on how to run each model can be found on our [website](http://nlp_architect.nervanasys.com/).

## NLP Architect library design philosophy

NLP Architect aspires to enable quick development of state-of-art NLP/NLU algorithms and to showcase Intel AI's efforts in deep-learning software optimization (Tensorflow MKL-DNN, etc.)
The library is designed around the life cycle of model development - pre-process, build model, train, validate, infer, save or deploy.

The main design guidelines are:

* Deep Learning framework agnostic
* Develop topologies utilized in NLP models
* NLP/NLU models implementation using included topologies
* Showcase End-to-End applications (Solutions) utilizing one or more NLP Architect model
* Generic dataset loaders, textual data processing utilities, and miscellaneous utilities that support NLP model development (loaders, text processors, io, metrics, etc.)
* Pythonic API for training and inference
* REST API servers with ability to serve trained models via HTTP
* Extensive model documentation and tutorials

## Demo UI examples

Dependency parser
<p>
  <img src="https://raw.githubusercontent.com/NervanaSystems/nlp-architect/master/assets/bist-demo-small.png" height="375"/>
</p>
Intent Extraction
<p>
  <img src="https://raw.githubusercontent.com/NervanaSystems/nlp-architect/master/assets/ie-demo-small.png" height="375"/>
<p>

## Packages

| Package                 	| Description                                          	|
|-------------------------	|------------------------------------------------------	|
| `nlp_architect.api`      	| Model server API interfaces                          	|
| `nlp_architect.common`   	| Common packages                                      	|
| `nlp_architect.contrib`  	| Framework extensions                                 	|
| `nlp_architect.data`     	| Datasets, data loaders and data classes              	|
| `nlp_architect.models`   	| NLP, NLU and End-to-End neural models                	|
| `nlp_architect.pipelines`	| End-to-end NLP apps                                  	|
| `nlp_architect.server`   	| API Server and demos UI                              	|
| `nlp_architect.solutions` | Solution applications                                	|
| `nlp_architect.utils`    	| Misc. I/O, metric, pre-processing and text utilities 	|

### Note
NLP Architect is an active space of research and development; Throughout future
releases new models, solutions, topologies and framework additions and changes
will be made. We aim to make sure all models run with Python 3.6+. We
encourage researchers and developers to contribute their work into the library.

## Citing

If you use NLP Architect in your research, please use the following citation:
```
@misc{izsak_peter_2018_1477518,
  author       = {Izsak, Peter and
                  Bethke, Anna and
                  Korat, Daniel and
                  Yaccobi, Amit and
                  Mamou, Jonathan and
                  Guskin, Shira and
                  Nittur Sridhar, Sharath and
                  Keller, Andy and
                  Pereg, Oren and
                  Eirew, Alon and
                  Tsabari, Sapir and
                  Green, Yael and
                  Kothapalli, Chinnikrishna and
                  Eavani, Harini and
                  Wasserblat, Moshe and
                  Liu, Yinyin and
                  Boudoukh, Guy and
                  Zafrir, Ofir and
                  Tewani, Maneesh},
  title        = {NLP Architect by Intel AI Lab},
  month        = nov,
  year         = 2018,
  doi          = {10.5281/zenodo.1477518},
  url          = {https://doi.org/10.5281/zenodo.1477518}
}
```

## Disclaimer
The NLP Architect is released as reference code for research purposes. It is
not an official Intel product, and the level of quality and support may not be
as expected from an official product. NLP Architect is intended to be used
locally and has not been designed, developed or evaluated for production
usage or web-deployment. Additional algorithms and environments are planned
to be added to the framework. Feedback and contributions from the open source
and NLP research communities are more than welcome.

## Contact
Contact the NLP Architect development team through Github issues or
email: nlp_architect@intel.com

[documentation]:http://nlp_architect.nervanasys.com
[Intel-Optimized TensorFlow with MKL-DNN]:https://software.intel.com/en-us/articles/intel-optimized-tensorflow-wheel-now-available
[TensorFlow]:https://www.tensorflow.org/
[Dynet]:https://dynet.readthedocs.io/en/latest/
