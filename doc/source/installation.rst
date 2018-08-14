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
  The default installation of NLP Architect use CPU-based binaries of all deep learning frameworks. Please refer to each framework's website for GPU-based installation instructions.


Instructions
============

We recommend installing NLP Architect within a virtual environment to ensure a self-contained environment.
To install NLP Architect models within an already existing virtual environment, see below installation receipes for custom model installation.
The default installation will create a new local virtual environment for development purposes.

To get started using our library, clone our repository:

.. code:: python

  git clone https://github.com/NervanaSystems/nlp-architect.git
  cd nlp-architect

Note that the ``setuptools`` package from a recent version of ``pip`` is needed to get the ``make`` command to build properly.

.. code:: python

  pip3 install -U setuptools

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


*  System-wide install (might require `sudo` permissions):

.. code:: python

  make sysinstall

Installing Intel® optimized Tensorflow with MKL-DNN
---------------------------------------------------

Tensorflow has a guide `guide <https://www.tensorflow.org/install/install_sources>`_ for compiling and installing Tensorflow with with MKL-DNN optimization. Make sure to install all required tools: bazel and python development dependencies.

Alternatively, follow the instructions below to compile and install the latest version of Tensorflow with MKL-DNN:

* Clone Tensorflow repository from github:

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
