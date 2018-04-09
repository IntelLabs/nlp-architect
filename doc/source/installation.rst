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

Installation
############

Requirements
============

The Intel® NLP Architect requires **Python 3.5+** running on a
Linux* or UNIX-based OS.
Each model contains its own installation requirements, but in general,
before installing, ensure your system has recent updates of the following packages:

.. csv-table::
   :header: "Ubuntu* 16.04+ or CentOS* 7.4+", "Mac OS X*", "Description"
   :widths: 20, 20, 42
   :escape: ~

   python-pip, pip, Tool to install Python dependencies
   python-virtualenv (*), virtualenv (*), Allows creation of isolated environments ((*): This is required only for Python 2.7 installs. With Python3: test for presence of ``venv`` with ``python3 -m venv -h``)
   libhdf5-dev, h5py, Enables loading of hdf5 formats
   pkg-config, pkg-config, Retrieves information about installed libraries

.. note::
  The default installations of models use the default CPU backend installations. In order to use GPU/MKL backends the user must install it himself.

Instructions
============

We recommend installing NLP Architect within a virtual environment to ensure a self-contained environment. To install NLP Architect models within an already existing virtual environment, see below installation receipes for custom model installation. The default installation will create a new local virtual environment for development purposes.

To get started using our library, run the following commands:

.. code:: python

  git clone https://github.com/NervanaSystems/ai-lab-nlp.git
  cd ai-lab-nlp; make setup_env
