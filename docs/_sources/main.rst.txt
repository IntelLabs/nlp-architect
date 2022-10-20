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

==============================
NLP Architect by Intel® AI Lab
==============================

NLP Architect is an open source Python library for exploring state-of-the-art deep learning topologies and techniques for optimizing Natural Language Processing and Natural Language Understanding neural network.

The library includes our past and ongoing NLP research and development efforts as part of Intel AI Lab.

NLP Architect can be downloaded from Github: https://github.com/IntelLabs/nlp-architect

Overview
========

NLP Architect is designed to be flexible for adding new models, neural network components, data handling methods and for easy training and running models.

Features:

* Core NLP models used in many NLP tasks and useful in many NLP applications
* Novel NLU models showcasing novel *topologies* and *techniques*
* **Optimized NLP/NLU models** showcasing different optimization algorithms on neural NLP/NLU models
* Model-oriented design:

  * Train and run models from command-line.
  * API for using models for inference in python.
  * Procedures to define custom processes for training, inference or anything related to processing.
  * CLI sub-system for running procedures
  
* Based on the following Deep Learning frameworks:

  * TensorFlow
  * PyTorch
  * Intel-Optimized TensorFlow with MKL-DNN
  * Dynet

* Essential utilities for working with NLP models - Text/String pre-processing, IO, data-manipulation, metrics, embeddings.


Library design philosophy
=========================

NLP Architect is a model-oriented library designed to showcase novel and different neural network optimizations. The library contains NLP/NLU related models per task, different neural network topologies (which are used in models), procedures for simplifying workflows in the library, pre-defined data processors and dataset loaders and misc utilities. The library is designed to be a tool for model development: data pre-process, build model, train, validate, infer, save or load a model.

The main design guidelines are:

* Deep Learning framework agnostic
* NLP/NLU models per task
* Different topologies (moduels) implementations that can be used with models
* Showcase End-to-End applications (Solutions) utilizing one or more NLP Architect model
* Generic dataset loaders, textual data processing utilities, and miscellaneous utilities that support NLP model development (loaders, text processors, io, metrics, etc.)
* ``Procedures`` for defining processes for training, inference, optimization or any kind of elaborate script.
* Pythonic API for using models for inference
* Extensive model documentation and tutorials

Disclaimer
==========

NLP Architect is an active space of research and development; Throughout future releases new models, solutions, topologies and framework additions and changes will be made. We aim to make sure all models run with Python 3.6+. We encourage researchers and developers to contribute their work into the library.

.. _Tensorflow: https://www.tensorflow.org/
.. _Intel-Optimized TensorFlow: https://software.intel.com/en-us/articles/intel-optimized-tensorflow-wheel-now-available
.. _Dynet: https://dynet.readthedocs.io/en/latest/
