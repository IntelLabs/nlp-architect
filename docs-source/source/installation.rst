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

The NLP Architect requires **Python 3.6+** running on a
Linux* or UNIX-based OS (like Mac OS). We recommend using the library with Ubuntu 16.04+.
We recommand installing basic OS compilers and python development packages.

Before installing the library make sure you has the most recent packages listed below:

.. csv-table::
   :header: "Ubuntu* 16.04+ or CentOS* 7.4+", "Mac OS X*", "Description"
   :widths: 20, 20, 42
   :escape: ~

   python-pip, pip, Tool to install Python dependencies
   python-dev, python-dev, Python development dependencies
   libhdf5-dev, h5py, Enables loading of hdf5 formats
   pkg-config, pkg-config, Retrieves information about installed libraries

.. note::

    The installation of NLP Architect will install CPU-based binaries of all deep learning frameworks.
    For specific installation of backends of Tensorflow or PyTorch (CPU/MKL/GPU) we recommend installing NLP Architect and then installing the desired backend DL framework.


Prerequisites (creating virtual env)
------------------------------------

Make sure ``pip`` and ``setuptools`` and ``venv`` are up to date before installing.

.. code:: bash

    pip3 install -U pip setuptools venv

.. note::

    We recommend installing NLP Architect in a virtual environment to self-contain
    the work done using the library.
    Users can use any virtual environment (pip, conda, etc.) 

To create and activate a new virtual environment (or skip this step and use the wizard below):

.. code:: bash

    python3.6 -m venv <my_new_env>
    source <my_new_env>/bin/activate

Install from ``pip``
====================

.. code:: bash

    pip install nlp-architect

Install from source
===================

.. code:: bash

    git clone https://github.com/IntelLabs/nlp-architect.git
    cd nlp-architect
    pip install -e .


Running Examples and Solutions
==============================

To run provided examples and solutions please install the library with [all] flag which will install extra packages required. (requires installation from source)

.. code:: bash

    pip install .[all]


Updating NLP Architect
======================

Depending of how you installed NLP Architect to update the library:

Using ``pip``
-----------

.. code:: bash

    pip install -U nlp-architect

From source
-----------

.. code:: bash

    git pull origin master
