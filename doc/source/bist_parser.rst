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

BIST Dependency Parser
======================

Graph-based dependency parser using BiLSTM feature extractors
-------------------------------------------------------------

The techniques behind the parser are described in the paper `Simple and
Accurate Dependency Parsing Using Bidirectional LSTM Feature
Representations <https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198>`__.
Futher materials could be found
`here <http://elki.cc/#/article/Simple%20and%20Accurate%20Dependency%20Parsing%20Using%20Bidirectional%20LSTM%20Feature%20Representations>`__.

Installation
------------

.. code:: bash

    pip install -r requirements.txt

Dependencies:
-------------

-  **Python** 3.5+
-  **dynet** 2.0.2
-  **numpy** 1.14.0
-  **cython** 0.27.3

Usage
-----

The module can be imported and used in python or invoked from the
command line. To import in python type the following:

.. code:: python

    from libs.bist.model import BISTParser

Training
--------

The software requires having a ``train.conll`` and ``dev.conll`` files
formatted according to the `CoNLL data
format <http://universaldependencies.org/format.html>`__. The benchmark
was performed on a Mac book pro with i7 processor. The parser achieves
an accuracy of 93.8 UAS on the standard Penn Treebank dataset (Standford
Dependencies). The trained models include improvements beyond those
described in the paper, to be published soon.

To train a parsing model type the following:

**Python**

.. code:: python

    parser = BISTParser()
    model = parser.train(outdir='output/dir', train='train.conll', dev='dev.conll', epochs=30, lstmdims=125, lstmlayers=2, pembedding=25, activation=tanh)

**Command line**

.. code:: bash

    python train.py --outdir [output/dir] --train train.conll --dev dev.conll [--epochs 30] [--lstmdims 125] [--lstmlayers 2] [--pembedding 25] [--activation tanh]

**The training process produces the following files in the output
directory:**

- *params.pickle* - parameters file required for loading the model afterwards (should be in the same directory as the .model file)
- for each completed epoch, denoted by **n**:

  - *dev\_epoch\_n.conll* - prediction results on dev file after **n** iterations
  - *dev\_epoch\_n.conll.txt* - accuracy results on dev file after **n** iterations
  - *bist\_epoch\_n.model* - the generated model after **n** iterations

.. note::
  The arguments in brackets are optional (see example 1 below);
  their default values appear above. These are the values used for
  training the pre-trained model.

.. note::
  You can train without pos embeddings by specifying (--pembedding 0).

.. note::
  The reported test result is the one matching the highest development score.

.. note::
  The parser calculates the accuracies excluding punctuation symbols by running the ``eval.pl`` script from the CoNLL-X Shared Task

Example 1
~~~~~~~~~

**Python**

.. code:: python

    parser = BISTParser()
    model = parser.train(outdir='out', train='train.conll', dev='dev.conll')

**Command line**

.. code:: bash

      python train.py --outdir out --train train.conll --dev dev.conll

Example 2
~~~~~~~~~

**Python**

.. code:: python

    parser = BISTParser()
    model = parser.train(outdir='out', train='train.conll', dev='dev.conll', activation='relu', epochs=20)

**Command line**

.. code:: bash

    python train.py --outdir out --train train.conll --dev dev.conll --activation relu --epochs 20

Inference
---------

The input file for inference must be annotated with part-of-speech tags,
in the `CoNLL data
format <http://universaldependencies.org/format.html>`__.

To run inference on an input file with a previously trained model type
the following:

**Python**

.. code:: python

    parser = BISTParser()
    results = parser.inference(outdir='output/dir', input='input.conll', model='.model/file', eval=True)

**Command line**

.. code:: bash

    python inference.py --outdir [output/dir] --input input.conll --model [.model/file] [--eval]

**The inference process produces the following files in the output directory:**

- *inference\_res.conll* - prediction results on the input file
- *inference\_res.conll.txt* - accuracy results on achieved on input file (see Note 3 below)

.. note::
  The arguments in brackets are optional; their default values appear above.

.. note::
  The model file has to be in the same directory as the params.pickle file generated during its training.

.. note::
  Accuracy results are generated only if the ``eval`` flag was specified and the input file is annotated with dependencies. This evaluation is produced by the ``eval.pl`` script.

Example 1
~~~~~~~~~

**Python**

.. code:: python

    parser = BISTParser()
    trained_model = parser.train(outdir='out', train='train.conll', dev='dev.conll')
    results = parser.inference(model=trained_model, outdir='out', input='input.conll')

Example 2
~~~~~~~~~

**Python**

.. code:: python

    parser = BISTParser()
    results = parser.inference(model='bist_epoch_1.model', outdir='out', input='input.conll')

**Command line**

.. code:: bash

    python inference.py --outdir out --input input.conll --model bist_epoch_1.model --eval

Citations
---------
* Kiperwasser, E., & Goldberg, Y. (2016). Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations. Transactions Of The Association For Computational Linguistics, 4, 313-327. https://transacl.org/ojs/index.php/tacl/article/view/885/198
