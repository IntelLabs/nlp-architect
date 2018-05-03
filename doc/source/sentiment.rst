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

Sentiment Solution
###################

Overview
========

The sentiment analysis solution is divided into two main modules:

1. Sentiment lexicon acquisition module
2. Sentiment classification module.

The goal of the first module is to acquire sentiment terms and aspect terms and produce the sentiment lexicon and aspect lexicon.
The goal of the inference module is to use the acquired lexicons to classify the sentiment in the corpus and visualize the extracted sentiment. The analysis includes aspect level statistics and sentiment event examples.

The following is a high-level diagram of the sentiment analysis solution:

.. image :: ../../solutions/sentiment/solution_block_diagram.png

UI description
===============
The user interface is divided to 3 main views (in a single one page):
1. “aspect sentiment bar chart” - is a bar chart that displays the of number of positive and negative sentiment events per aspect (sentiment target) or the accumulated sentiment score per aspect.
2. “Aspect selection checklist”: in this checklist the user is able to select the aspects that are be shown in the “aspect sentiment bar chart” at the top of the screen.
3. “Sentiment events view”: upon double clicking on specific aspect bar in the aspect sentiment bar chart - this view displays the 5 most positive/negative sentiment events towards this aspect.

Following is a screenshot of the demo UI:

.. image :: ../../solutions/sentiment/UI_screen.png

Installation
===============
To install all dependencies, run the included **install.sh** script:

.. code:: python

  chmod +x install.sh
  ./install.sh

Usage
=======
The module needs to be imported by typing the following:

.. code:: python

  from sentiment_analysis.sentiment_training.train import SentimentTraining
  from sentiment_analysis.sentiment_inference.inference import SentimentInference

.. note:: To import, the root directory ``/ai-lab-models`` must be added to ``PYTHONPATH`` environment variable.

Files
=======
**sentiment_training**

- **train.py** : main training execution module
- **term_extraction/ExtractTerms-1.0.jar** : candidate terms extraction algorithm
- **term_reranking/term_reranking.py** : term re-ranking component

**sentiment_inference**

- **inference.py** : main inference execution module
- **lexical_sentiment_classification/execute/lexical_sentiment_classification.py** : classifies aspects sentiment in given corpus using acquired lexicons
- **statistical_analysis/execute/execute_statistical_analysis.py** : calculates statistical analysis on classifiers output
- **ui/sentiment_ui.py** :visualize the extracted sentiment


Dataset
=======
The dataset consists of two parts:

1. Corpus - a directory of raw text files
2. Aspect file - A csv file of aspects, where each row contains the different terms describing an aspect

Training
=========
The training module accepts as input a dataset in the form of a directory of raw text files.
The output is a dictionary of detected terms with their predicted polarities and certainty scores.

.. code:: python

  train = SentimentTraining()
  terms = train.run('/path/to/dataset')

Or within the command line:

.. code:: python

  python train.py --corpus_dir <directory to corpus>

Inference
=========
The inference module accepts as input:

1. A dataset in the form of a directory of raw text files
2. A csv file of aspects, where each row contains the different terms describing an aspect

The final output is a web UI presenting a bar chart and examples of the aspects in the corpus.
A detailed json containing all sentiment events in the corpus is outputted as well.

.. code:: python

  inference = SentimentInference()
  inference.run('/path/to/dataset', '/path/to/aspect/file')

Or within command line:

.. code:: python

  python inference.py --corpus_dir <directory to corpus> --aspect_file <aspect file>

Results
=========
The following is a benchmark of sentiment term extraction measured on a 2000 document corpus related to business news domain:

.. image :: ../../solutions/sentiment/Benchmark.png

Citation
=========
Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations: https://transacl.org/ojs/index.php/tacl/article/view/885

Opinion Word Expansion and Target Extraction through Double Propagation: https://www.mitpressjournals.org/doi/pdf/10.1162/coli_a_00034
