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

Installation
############

Requirements
============

The NLP Architect requires **Python 3.5+** running on a
Linux* or UNIX-based OS (like Mac OS). We recommend using the library with Ubuntu 16.04+.

Before installing the library make sure you has the most recent packages listed below:

.. csv-table::
   :header: "Ubuntu* 16.04+ or CentOS* 7.4+", "Mac OS X*", "Description"
   :widths: 20, 20, 42
   :escape: ~

   python-pip, pip, Tool to install Python dependencies
   python-virtualenv (*), virtualenv (*), Allows creation of isolated environments ((*): This is required only for Python 2.7 installs. With Python3: test for presence of ``venv`` with ``python3 -m venv -h``)
   libhdf5-dev, h5py, Enables loading of hdf5 formats
   pkg-config, pkg-config, Retrieves information about installed libraries

.. note::
  The default installations of NLP Architect uses the default CPU backend of all deep learning framework installations. When installing on Ubuntu systems optional Intel optimized frameworks can be approved to be installed.

Instructions
============

We recommend installing NLP Architect within a virtual environment to ensure a self-contained environment.
To install NLP Architect models within an already existing virtual environment, see below installation receipes for custom model installation.
The default installation will create a new local virtual environment for development purposes.

To get started using our library, clone our repository:

.. code:: python

  git clone https://github.com/NervanaSystems/nlp-architect.git
  cd nlp-architect

Installing within a virtual environment
---------------------------------------

*  Install in development mode (default):

.. code:: python

  make

*  Complete install:

.. code:: python

  make install

*  Activate the newly created virtual environment:

.. code:: python

  . .nlp_architect_env/bin/activate

* Fire up your favorite IDE/text editor/terminal and start running models

Installing to current working python (or system wide install)
-------------------------------------------------------------

*  Install without creating a new virtual environment:

.. code:: python

  make install_no_virt_env
