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
   libhdf5-dev, h5py, Enables loading of hdf5 formats
   pkg-config, pkg-config, Retrieves information about installed libraries

.. note::
  The default installation of NLP Architect use CPU-based binaries of all deep learning frameworks. Intel Optimized MKL-DNN binaries will be installed if a Linux is detected. GPU backed is supported online on Linux and if a GPU is present. See details below for instructions on how to install each backend.


Installation
============
Prerequisites
-------------

Make sure ``pip`` and ``setuptools`` and ``venv`` are up to date before installing.

.. code:: bash

    pip3 install -U pip setuptools venv

We recommend installing NLP Architect in a virtual environment to self-contain
the work done using the library.

To create and activate a new virtual environment (or skip this step and use the wizard below):

.. code:: bash

    python3 -m venv .nlp_architect_env
    source .nlp_architect_env/bin/activate


.. include:: _quick_install.rst

Install from source
-------------------

To get started, clone our repository:

.. code:: bash

    git clone https://github.com/NervanaSystems/nlp-architect.git
    cd nlp-architect

Selecting a backend
^^^^^^^^^^^^^^^^^^^

NLP Architect supports CPU, GPU and Intel Optimized Tensorflow (MKL-DNN) backends.
Users can select the desired backend using a dedicated environment variable (default: CPU). (MKL-DNN and GPU backends are supported only on Linux)

.. code:: bash

    export NLP_ARCHITECT_BE=CPU/MKL/GPU

Installation
^^^^^^^^^^^^

NLP Architect is installed using `pip` and it is recommended to install in development mode.

Default:

.. code:: bash

    pip3 install .

Development mode:

.. code:: bash

    pip3 install -e .

Once installed, the ``nlp_architect`` command provides additional options to work with the library, issue ``nlp_architect -h`` to see all options.

======

Compiling Intel® optimized Tensorflow with MKL-DNN
==================================================

NLP Architect supports MKL-DNN flavor of Tensorflow out of the box, however, if the user wishes to compile Tensorflow we provide instructions below.

Tensorflow has a guide `guide <https://www.tensorflow.org/install/install_sources>`_ for compiling and installing Tensorflow with with MKL-DNN optimization. Make sure to install all required tools: bazel and python development dependencies.

Alternatively, follow the instructions below to compile and install the latest version of Tensorflow with MKL-DNN:

* Clone Tensorflow repository from GitHub:

  .. code::

      git clone https://github.com/tensorflow/tensorflow
      cd tensorflow

* Configure Tensorflow for compilation:

  .. code::

      ./configure

* Compile Tensorflow with MKL-DNN:

  .. code::

      bazel build --config=mkl --config=opt //tensorflow/tools/pip_package:build_pip_package

* Create pip package in ``/tmp/tensorflow_pkg``:

  .. code::

      bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

* Install Tensorflow pip package:

  .. code::

      pip install <tensorflow package name>.whl

* Refer to `this <https://www.tensorflow.org/performance/performance_guide#tensorflow_with_intel_mkl_dnn>`_ guide for specific configuration to get optimal performance when running your model.
