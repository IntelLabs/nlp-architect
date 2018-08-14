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

Noun Phrase Semantic Segmentation
###################################

Overview
========
Noun-Phrase (NP) is a phrase which has a noun (or pronoun) as its head and zero of more dependent modifiers.
Noun-Phrase is the most frequently occurring phrase type and its inner segmentation is critical for understanding the
semantics of the Noun-Phrase.
The most basic division of the semantic segmentation is to two classes:

1. Descriptive Structure - a structure where all dependent modifiers are not changing the semantic meaning of the Head.
2. Collocation Structure - a sequence of words or term that co-occur and change the semantic meaning of the Head.

For example:

- ``fresh hot dog`` - hot dog is a collocation, and changes the head (``dog``) semantic meaning.
- ``fresh hot pizza`` - fresh and hot are descriptions for the pizza.

Model
=====

The :py:class:`NpSemanticSegClassifier <nlp_architect.models.np_semantic_segmentation.NpSemanticSegClassifier>` model is the first step in the Semantic Segmentation algorithm - the MLP classifier.
The Semantic Segmentation algorithm takes the dependency relations between the Noun-Phrase words, and the MLP classifier inference as the
input - and build a semantic hierarchy that represents the semantic meaning.
The Semantic Segmentation algorithm eventually create a tree where each tier represent a semantic meaning -> if a sequence of words is a
collocation then a collocation tier is created, else the elements are broken down and each one is mapped
to different tier in the tree.

This model trains MLP classifier and inference from such classifier in order to conclude the correct segmentation
for the given NP.

For the examples above the classifier will output 1 (==Collocation) for ``hot dog`` and output 0 (== not collocation)
for ``hot pizza``.

Files
=====
- :py:class:`NpSemanticSegClassifier <nlp_architect.models.np_semantic_segmentation.NpSemanticSegClassifier>`: is the MLP classifier model.
- **examples/np_semantic_segmentation/data.py**: Prepare string data for both ``train.py`` and ``inference.py`` using pre-trained word embedding, NLTKCollocations score, Wordnet and wikidata.
- **examples/np_semantic_segmentation/feature_extraction.py**: contains the feature extraction services
- **examples/np_semantic_segmentation/train.py**: train the MLP classifier.
- **examples/np_semantic_segmentation/inference.py**: load the trained model and inference the input data by the model.

Dataset
=======
The expected dataset is a CSV file with 2 columns. the first column
contains the Noun-Phrase string (a Noun-Phrase containing 2 words), and
the second column contains the correct label (if the 2 word Noun-Phrase
is a collocation - the label is 1, else 0)

If you wish to use an existing dataset for training the model, you can
download Tratz 2011 et al. dataset [1]_ [2]_ [3]_ [4]_ from the following link: `Tratz
2011
Dataset <https://vered1986.github.io/papers/Tratz2011_Dataset.tar.gz>`__.
Is also available in
`here <https://www.isi.edu/publications/licensed-sw/fanseparser/index.html>`__.
(The terms and conditions of the data set license apply. Intel does not
grant any rights to the data files or database.

After downloading and unzipping the dataset, run
``preprocess_tratz2011.py`` in order to construct the labeled data and
save it in a CSV file (as expected for the model). The scripts read 2
.tsv files ('tratz2011\_coarse\_grained\_random/train.tsv' and
'tratz2011\_coarse\_grained\_random/val.tsv') and outputs 2 .csv files
accordingly to the same location.

Quick example:

::

    python examples/np_semantic_segmentation/preprocess_tratz2011.py --data path_to_Tratz_2011_dataset_folder

Pre-processing the data
-----------------------

A feature vector is extracted from each Noun-Phrase string using the
command ``python data.py``

-  Word2Vec word embedding (300 size vector for each word in the
   Noun-Phrase) .

   -  Pre-trained Google News Word2vec model can download
      `here <https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing>`__
   -  The terms and conditions of the data set license apply. Intel does
      not grant any rights to the data files or database.

-  Cosine distance between 2 words in the Noun-Phrase.
-  NLTKCollocations score (PMI score (from Manning and Schutze 5.4) and Chi-square score (Manning and Schutze 5.3.3)).
-  A binary features whether the Noun-Phrase has existing entity in
   Wikidata.
-  A binary features whether the Noun-Phrase has existing entity in
   WordNet.

Quick example:

.. code:: python

  python data.py --data input_data_path.csv --output output_prepared_path.csv --w2v_path <path_to_w2v>/GoogleNews-vectors-negative300.bin.gz

Running Modalities
==================
Training
--------

The command ``python examples/np_semantic_segmentation/train.py`` will train the MLP classifier and
evaluate it. After training is done, the model is saved automatically:

Quick example:

.. code:: python

  python examples/np_semantic_segmentation/train.py \
    --data prepared_data_path.csv \
    --model_path np_semantic_segmentation_path.h5

Inference
---------

In order to run inference you need to have pre-trained
``<model_name>.h5`` & ``<model_name>.json`` files and data CSV file that was generated by
``prepare_data.py``. The result of ``python inference.py`` is a CSV
file, each row contains the model's inference in respect to the input
data.

Quick example:

.. code:: python

  python examples/np_semantic_segmentation/inference.py \
    --model np_semantic_segmentation_path.prm \
    --data prepared_data_path.csv \
    --output inference_data.csv \
    --print_stats

References
==========

.. [1] Stephen Tratz and Eduard Hovy. 2011. A Fast, Accurate, Non-Projective, Semantically-Enriched Parser. In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing. Edinburgh, Scotland, UK.
.. [2] Dirk Hovy, Stephen Tratz, and Eduard Hovy. 2010. What’s in a Preposition? Dimensions of Sense Disambiguation for an Interesting Word Class. In Proceedings of COLING 2010: Poster Volume. Beijing, China.
.. [3] Stephen Tratz and Dirk Hovy. 2009. Disambiguation of Preposition Sense using Linguistically Motivated Features. In Proceedings of Human Language Technologies: The 2009 Annual Conference of the North American Chapter of the Association for Computational Linguistics, Companion Volume: Student Research Workshop and Doctoral Consortium. Boulder, Colorado.
.. [4] Stephen Tratz and Eduard Hovy. 2010. A Taxonomy, Dataset, and Classifier for Automatic Noun Compound Interpretation. In Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics. Uppsala, Sweden
