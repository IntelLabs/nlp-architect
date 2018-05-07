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

NER Expansion
##############

Overview
========
Named entity recognition (NER) consists in locating and classifying named entities in text into
predefined categories such as the names of persons, organizations, locations, expressions of times,
quantities, monetary values, percentages, etc. The main drawback of the state-of-the-art systems is
that they require a large amount of data annotated with a predefined set of categories in order to
train statistical models. These models are trained per domain and per set of categories.

Named entity set expansion refers to expanding a given partial set of named entity into a more
complete set. We are interested in extracting set of named-entities appearing in a target corpus
without any prior knowledge on the corpus and on the set of categories. Given a seed with few named
entity examples belonging to the same implicit category, we use word embedding's in order to generate
set of named-entities from the same category. For example, given Siri and Cortana as seed in a corpus
on Artificial Intelligence, one might expect to get other assistant products.

Our approach takes advantage of the word embedding's in order to expand. Word embedding's attempts
to understand meaning and semantic relationships among words. It works in a way that is similar to
deep approaches, such as recurrent neural networks or deep neural networks, but is computationally
more efficient. Word embedding's reflect conceptual association or relatedness between the named entities.

The training process is as follows: given few named entity examples (the seed), we enrich their
word embedding's with additional features: case of the first letter (if relevant) and, for each
category, the cosine distance between the centroid of the seed and the named entity word embedding's.
If negative samples are not provided, we randomly sample them. Then we train an MLP classifier.

The inference process is as follows

1) The system finds the most similar named entities in the vector space model according to the cosine similarity
2) We enrich the word embedding's with additional features as during training and we classify the named entities returned by cosine similarity.

Dependencies
============
- **gensim** (in prepare_data.py, used for basic word embedding's utilities)

Running Modalities
==================

Prepare training, validation and test sets
------------------------------------------

.. code:: python

  python prepare_data [--seed_length SEED_LENGTH]
         [--word_embedding_model_file WORD_EMBEDDING_MODEL_FILE]
         [--word_embedding_type WORD_EMBEDDING_TYPE] [--seed_file SEED_FILE]
         [--lower] [--ignore_phrase] [--data_set_file DATA_SET_FILE]

**Example 1:**

Expanding news related categories (capital, country, currency, airlines, newspapers, ice_hockey,
basketball, city, state) using pretrained Google News Word2vec model.

Pretrained Google News Word2vec model can download here_. The data news_categories.csv has been
created from https://github.com/dav/word2vec/tree/master/data.

.. code:: python

  python prepare_data.py --seed_length 10 --word_embedding_model_file pretrained_models/GoogleNews-vectors-negative300.bin \
  --word_embedding_type word2vec --seed_file data/news_categories.csv`

**Example 2:**

Expanding news related categories (states, capitals) using pretrained Google News Word2vec model

The data state_capital_categories.csv has been created from https://en.wikipedia.org/wiki/List_of_capitals_in_the_United_States.

.. code:: python

  python prepare_data.py --seed_length 10 --word_embedding_model_file pretrained_models/GoogleNews-vectors-negative300.bin \
  --word_embedding_type word2vec --seed_file data/state_capital_categories.csv

**Example 3:**

Expanding programming language category using pretrained FastText English model

Pretrained FastText English model wiki.en.bin can download from AWS_. The data
state_capital_categories.csv is the same as Example 2.

.. code:: python

  python prepare_data.py --seed_length 10 --word_embedding_model_file pretrained_models/wiki.en.bin \
  --word_embedding_type fasttext --seed_file data/programming_languages_categories.csv \
  --lower --ignore_phrase

Training
--------
Train the MLP classifier and evaluate it.

.. code:: python

  python train.py [--data_set_file DATA_SET_FILE] [--model_prm MODEL_PRM]

Inference
---------
The trained MLP classifier is used for inference. The top N results are logged for both methods:
cosine distance and MLP classification.

.. code:: python

  python inference.py [--N N] [--data_set_file DATA_SET_FILE] [--model_prm MODEL_PRM]


.. _here: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
.. _AWS: https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip
