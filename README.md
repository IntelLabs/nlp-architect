<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/IntelLabs/nlp-architect/master/docs-source/source/assets/nlp_architect_logo_trans.png" width="400"/>
    <br>
<p>
<h2 align="center">
A Deep Learning NLP/NLU library by <a href="https://www.intel.ai/research/">Intel® AI Lab</a>
</h2>
<p align="center">
    <a href="https://github.com/IntelLabs/nlp-architect/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/IntelLabs/nlp-architect.svg?color=blue&style=flat-square">
    </a>
    <a href="https://intellabs.github.io/nlp-architect">
        <img alt="Website" src="https://img.shields.io/website/http/intellabs.github.io/nlp-architect.svg?down_color=red&down_message=offline&style=flat-square&up_message=online">
    </a>
    <a href="https://github.com/IntelLabs/nlp-architect/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/IntelLabs/nlp-architect.svg?style=flat-square">
    </a>
    <a href="https://pepy.tech/project/nlp-architect">
        <img src="https://pepy.tech/badge/nlp-architect">
    </a>
</p>

<h4 align="center">
  <a href="#overview">Overview</a> |
  <a href="#models">Models</a> |
  <a href="#installing-nlp-architect">Installation</a> |
  <a href="https://github.com/IntelLabs/nlp-architect/tree/master/examples">Examples</a> <a href="https://intellabs.github.io/nlp-architect/"></a> |
  <a href="https://intellabs.github.io/nlp-architect">Documentation</a> |
  <a href="https://github.com/IntelLabs/nlp-architect/tree/master/tutorials">Tutorials</a> |
  <a href="https://intellabs.github.io/nlp-architect/developer_guide.html">Contributing</a>
</h4>

NLP Architect is an open source Python library for exploring state-of-the-art
deep learning topologies and techniques for optimizing Natural Language Processing and
Natural Language Understanding Neural Networks.

## Overview

NLP Architect is an NLP library designed to be flexible, easy to extend, allow for easy and rapid integration of NLP models in applications and to showcase optimized models.

Features:

* Core NLP models used in many NLP tasks and useful in many NLP applications
* Novel NLU models showcasing novel topologies and techniques
* Optimized NLP/NLU models showcasing different optimization algorithms on neural NLP/NLU models
* Model-oriented design:
  * Train and run models from command-line.
  * API for using models for inference in python.
  * Procedures to define custom processes for training,    inference or anything related to processing.
  * CLI sub-system for running procedures
* Based on optimized Deep Learning frameworks:

  * [TensorFlow]
  * [PyTorch]
  * [Dynet]

* Essential utilities for working with NLP models - Text/String pre-processing, IO, data-manipulation, metrics, embeddings.

## Installing NLP Architect

We recommend to install NLP Architect in a new python environment, to use python 3.6+ with up-to-date `pip`, `setuptools` and `h5py`.

### Install using `pip`

Install core library only

```sh
pip install nlp-architect
```

### Install from source (Github)

Includes core library, examples, solutions and tutorials:

```sh
git clone https://github.com/IntelLabs/nlp-architect.git
cd nlp-architect
pip install -e .  # install in developer mode
```

### Running Examples and Solutions

To run provided examples and solutions please install the library with `[all]` flag which will install extra packages required. (requires installation from source)

```sh
pip install .[all]
```


## Models

NLP models that provide best (or near) in class performance:

* [Word chunking](https://intellabs.github.io/nlp-architect/tagging/sequence_tagging.html#word-chunker)
* [Named Entity Recognition](https://intellabs.github.io/nlp-architect/tagging/sequence_tagging.html#named-entity-recognition)
* [Dependency parsing](https://intellabs.github.io/nlp-architect/bist_parser.html)
* [Intent Extraction](https://intellabs.github.io/nlp-architect/intent.html)
* [Sentiment classification](https://intellabs.github.io/nlp-architect/sentiment.html#supervised-sentiment)
* [Language models](https://intellabs.github.io/nlp-architect/lm.html#language-modeling-with-tcn)
* [Transformers](https://intellabs.github.io/nlp-architect/transformers.html) (for NLP tasks)

Natural Language Understanding (NLU) models that address semantic understanding:

* [Aspect Based Sentiment Analysis (ABSA)](https://intellabs.github.io/nlp-architect/absa.html)
* [Joint intent detection and slot tagging](https://intellabs.github.io/nlp-architect/intent.html)
* [Noun phrase embedding representation (NP2Vec)](https://intellabs.github.io/nlp-architect/np2vec.html)
* [Most common word sense detection](https://intellabs.github.io/nlp-architect/word_sense.html)
* [Relation identification](https://intellabs.github.io/nlp-architect/identifying_semantic_relation.html)
* [Cross document coreference](https://intellabs.github.io/nlp-architect/cross_doc_coref.html)
* [Noun phrase semantic segmentation](https://intellabs.github.io/nlp-architect/np_segmentation.html)

Optimizing NLP/NLU models and misc. optimization techniques:

* [Quantized BERT (8bit)](https://intellabs.github.io/nlp-architect/quantized_bert.html)
* [Knowledge Distillation using Transformers](https://intellabs.github.io/nlp-architect/transformers_distillation.html)
* [Sparse and Quantized Neural Machine Translation (GNMT)](https://intellabs.github.io/nlp-architect/sparse_gnmt.html)

Solutions (End-to-end applications) using one or more models:

* [Term Set expansion](https://intellabs.github.io/nlp-architect/term_set_expansion.html) - uses the included word chunker as a noun phrase extractor and NP2Vec to create semantic term sets
* [Topics and trend analysis](https://intellabs.github.io/nlp-architect/trend_analysis.html) - analyzing trending phrases in temporal corpora
* [Aspect Based Sentiment Analysis (ABSA)](https://intellabs.github.io/nlp-architect/absa_solution.html)

## Documentation

Full library [documentation](https://intellabs.github.io/nlp-architect/) of NLP models, algorithms, solutions and instructions
on how to run each model can be found on our [website](https://intellabs.github.io/nlp-architect/).

## NLP Architect library design philosophy

NLP Architect is a _model-oriented_ library designed to showcase novel and different neural network optimizations. The library contains NLP/NLU related models per task, different neural network topologies (which are used in models), procedures for simplifying workflows in the library, pre-defined data processors and dataset loaders and misc utilities.
The library is designed to be a tool for model development: data pre-process, build model, train, validate, infer, save or load a model.

The main design guidelines are:

* Deep Learning framework agnostic
* NLP/NLU models per task
* Different topologies used in models
* Showcase End-to-End applications (Solutions) utilizing one or more NLP Architect model
* Generic dataset loaders, textual data processing utilities, and miscellaneous utilities that support NLP model development (loaders, text processors, io, metrics, etc.)
* Procedures for defining processes for training, inference, optimization or any kind of elaborate script.
* Pythonic API for using models for inference
* Extensive model documentation and tutorials

### Note

NLP Architect is an active space of research and development; Throughout future
releases new models, solutions, topologies and framework additions and changes
will be made. We aim to make sure all models run with Python 3.6+. We
encourage researchers and developers to contribute their work into the library.

## Citing

If you use NLP Architect in your research, please use the following citation:

    @misc{izsak_peter_2018_1477518,
      title        = {NLP Architect by Intel AI Lab},
      month        = nov,
      year         = 2018,
      doi          = {10.5281/zenodo.1477518},
      url          = {https://doi.org/10.5281/zenodo.1477518}
    }

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

[documentation]:https://intellabs.github.io/nlp-architect
[TensorFlow]:https://www.tensorflow.org/
[PyTorch]:https://pytorch.org/
[Dynet]:https://dynet.readthedocs.io/en/latest/
