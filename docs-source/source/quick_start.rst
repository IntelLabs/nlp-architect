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

===========
Quick start
===========

Installation
------------

Make sure to use **Python 3.6+** and a virtual environment.

Using ``pip``
~~~~~~~~~~~~~

.. code:: bash

    pip install nlp_architect

From source
~~~~~~~~~~~

.. code:: bash

    git clone https://github.com/IntelLabs/nlp-architect.git
    cd nlp-architect
    pip install -e .  # install in development mode

.. note::

    For specific installation of backends of Tensorflow or PyTorch (CPU/MKL/GPU) we recommend installing NLP Architect and then installing the desired package of framework.

Running Examples and Solutions
------------------------------
To run provided examples and solutions please install the library with [all] flag which will install extra packages required. (requires installation from source)

.. code:: bash

    pip install .[all]

Usage
-----

NLP Architect has the following packages:

+---------------------------+-------------------------------------------------------+
| Package                   | Description                                           |
+===========================+=======================================================+
| `nlp_architect.api`       | Model API interfaces                                  |
+---------------------------+-------------------------------------------------------+
| `nlp_architect.common`    | Common packages                                       |
+---------------------------+-------------------------------------------------------+
| `nlp_architect.cli`       | Command line module                                   |
+---------------------------+-------------------------------------------------------+
| `nlp_architect.data`      | Datasets, loaders and data processors                 |
+---------------------------+-------------------------------------------------------+
| `nlp_architect.models`    | NLP, NLU and End-to-End models                        |
+---------------------------+-------------------------------------------------------+
| `nlp_architect.nn`        | Topology related models and additions (per framework) |
+---------------------------+-------------------------------------------------------+
| `nlp_architect.pipelines` | End-to-end NLP apps                                   |
+---------------------------+-------------------------------------------------------+
| `nlp_architect.procedures`| Procedure scripts                                     |
+---------------------------+-------------------------------------------------------+
| `nlp_architect.utils`     | Misc. I/O, metric, pre-processing and text utilities  |
+---------------------------+-------------------------------------------------------+


CLI
---

NLP Architect comes with a CLI application that helps users run procedures and processes from the library.

.. warning::

    The CLI is in development and some functionality is not complete
    and will be added in future versions

NLP Architect commands:

.. code-block:: text

    nlp-train       Train a model from the library
    nlp-inference   Run a model from the library

Use ``nlp-train/inference -h`` for per command usage instructions.