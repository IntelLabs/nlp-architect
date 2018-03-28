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

Most Probable Word Sense
########################

Overview
========
The most common word sense algorithm's goal is to extract the most common sense of a target word.
The input to the algorithm is the target word and the output are the senses of the target word where
each sense is scored according to the most commonly used sense in the language.
note that most of the words in the language have many senses. The sense of a word a consists of the
definition of the word and the inherited hypernyms of the word.

For example: the most common sense of the target_word **burger** is:
``definition: "a sandwich consisting of a fried cake of minced beef served on a bun, often with other ingredients"
inherited hypernyms: ['sandwich', 'snack_food']`` whereas the least common sense is:
``definition: "United States jurist appointed chief justice of the United States Supreme Court by Richard Nixon (1907-1995)"``

Our approach:

**Training**: the training inputs a list of target_words where each word is associated with a correct (true example)
or incorrect (false example) sense. The sense consists of the definition and the inherited hypernyms
of the target word in a specific sense.

**Inference**: extracts all the possible senses for a specific target_word and scores those senses according
to the most common sense of the target_word. the higher the score the higher the probability of the sense being the most commonly used sense.

In both training and inference a feature vector is constructed as input to the neural network.
The feature vector consists of:

- the word embedding distance between the target_word and the inherited hypernyms
- 2 variations of the word embedding distance between the target_word and the definition
- the word embedding of the target_word
- the CBOW word embedding of the definition

Dependencies
============
The training module requires a gold standard csv file which is list of target_words where each word
is associated with a CLASS_LABEL - a correct (true example) or an incorrect (false example) sense.
The sense consists of the definition and the inherited hypernyms of the target word in a specific sense.
The user needs to prepare this gold standard csv file in advance.
The file should include the following 4 columns:

|TARGET_WORD|DEFINITION|SEMANTIC_BRANCH|CLASS_LABEL

where:

1. TARGET_WORD: the word that you want to get the most common sense of.
2. DEFINITION: the definition of the word (usually a single sentence) extracted from external resource such as wordnet or wikidata
3. SEMANTIC_BRANCH:  the inherited hypernyms extracted from external resource such as wordnet or wikidata
4. CLASS_LABEL: a binary [0,1] Y value that represent whether the sense (Definition and semantic branch) is the most common sense  of the target word

Store the file in the data folder of the project.
The code dependencies for this project are as follows:

- python version 3.6.3 was used in this project
- **nervananeon** (version 2.4.0 is used in this project)
- **aeon** (version 2.0.2 is used in this project)
- **gensim** (version 3.2.0. in prepare_data.py, used for basic word embedding's utilities)
- **nltk** (version 3.2.5 in Inference.py, used for extracting all senses of each word)
- **sklearn** (version 0.19.0. in prepare_data.py, used for splitting train and validation sets)
- **numpy**
- **csv** (in prepare_data.py, used for reading gold standard files)
- **codecs**
- **math**
- **pickle** (used for saving and reading data files)
- **termcolor** (in Inference.py, used for highlighting the detected sense)


Running Modalities
==================

Dataset Preparation
--------------------

The script prepare_data.py uses the gold standard csv file as described in the requirements section above
using pretrained Google News Word2vec model. Pretrained Google News Word2vec model can be download here_.

.. code:: python

  python prepare_data.py --gold_standard_file data/gold_standard.csv
       --word_embedding_model_file pretrained_models/GoogleNews-vectors-negative300.bin
       --training_to_validation_size_ratio 0.8
       --data_set_file data/data_set.pkl

Training
--------

Trains the MLP classifier and evaluate it.

.. code:: python

  python train.py --data_set_file data/data_set.pkl
                 --model_prm data/wsd_classification_model.prm

Inference
---------
.. code:: python

  python inference.py --max_num_of_senses_to_search 3
       --input_inference_examples_file data/input_inference_examples.csv
       --word_embedding_model_file pretrained_models/GoogleNews-vectors-negative300.bin
       --model_prm data/wsd_classification_model.prm

Where the ``max_num_of_senses_to_search`` is the maximum number of senses that are checked per target word (default =3)
and ``input_inference_examples_file`` is a csv file containing the input inference data. This file includes
a single column wherein each entry in this column is a different target word

.. note::
  The results are printed to the terminal using different colors therefore using a white terminal background is best to view the results

.. _here: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
