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

Intent Extraction
#################

Overview
========
Intent extraction is a type of Natural-Language-Understanding (NLU) task that helps to understand
the type of action conveyed in the sentences and all its participating parts.

An example of a sentence with intent could be:

**Siri, can you please remind me to pickup my laundry on my way home?**


The action conveyed in the sentence is to *remind* the speaker about something. The verb *remind*
applies that there is an assignee that has to do the action (who?), an assignee that the action
applies to (to whom?) and the object of the action (what?). In this case, *Siri* has to remind the
*speaker* to *pickup the laundry*.

Models
======
Multi-task Intent and slot tagging model
----------------------------------------

:py:class:`MultiTaskIntentModel <nlp_architect.models.intent_extraction.MultiTaskIntentModel>` is a Multi-task model that is similar to the joint intent/slot tagging model. The model has 2 sources of input: 1 - words, 2 - characters of words. The model has 3 main features when compared to the other models, character information embedding acting as a feature extractor of the words, a CRF classifier for slot labels, and a cascading structure of the intent and tag classification.
The intent classification is done by encoding the context of the sentences (words ``x_1, .., x_n``), using word embeddings (denoted as ``W``), by a bi-directional LSTM layer, and training a classifier on the last hidden state of the LSTM layer (using ``softmax``).
Word-character embeddings (denoted as ``C``) are created using a bi-directional LSTM encoder which concatenates the last hidden states of the layers.
The encoding of the word-context, in each time step (word location in the sentence) is concatenated with the word-character embeddings and pushed in another bi-directional LSTM which provides the final context encoding that a CRF layer uses for slot tag classification.

.. image :: assets/mtl_model.png

Encoder-Decoder topology for Slot Tagging
-----------------------------------------

The Encoder-Decoder LSTM topology is a well known model for sequence-to-sequence classification.
:py:class:`Seq2SeqIntentModel <nlp_architect.models.intent_extraction.Seq2SeqIntentModel>` is a model that is similar to the *Encoder-Labeler Deep LSTM(W)* model shown in [2]_.
The :py:class:`model <nlp_architect.models.intent_extraction.Seq2SeqIntentModel>` support arbitrary depths of LSTM layers in both encoder and decoder.

.. image :: assets/enc-dec_model.png

Datasets
========
SNIPS NLU benchmark
-------------------

A NLU benchmark [5]_ containing ~16K sentences with 7 intent types. Each intent has about 2000 sentences
for training the model and 100 sentences for validation. :py:class:`SNIPS <nlp_architect.data.intent_datasets.SNIPS>` is a class that loads the dataset from the repository and encodes the data into BIO format. The words are encoded with sparse int representation and word characters are extracted for character embeddings.

The dataset can be downloaded from https://github.com/snipsco/nlu-benchmark, and more info on the benchmark can be found here_. The terms and conditions of the data set license apply. Intel does not grant any rights to the data files.

Once the dataset is downloaded, use the following path ``<SNIPS folder>/2017-06-custom-intent-engines`` as the dataset path when using  :py:class:`SNIPS <nlp_architect.data.intent_datasets.SNIPS>` data loader.

TabularIntentDataset
--------------------
We provide an additional dataset loader  :py:class:`TabularIntentDataset <nlp_architect.data.intent_datasets.TabularIntentDataset>` which can parse tabular data in the format of:

-  each word encoded in a separate line: ``<token> <token_tag> <intent_type>``
-  sentences are separated with an empty line

The dataset loader extracts word and character sparse encoding and label/intent tags per sentence. This data-loader is useful for many intent extraction datasets that can be found on the web and used in academic literature (such as ATIS [3]_ [4]_, Conll, etc.).

Files
=====

- **examples/intent_extraction/train_enc-dec_model.py**: training script to train a :py:class:`Seq2SeqIntentModel <nlp_architect.models.intent_extraction.Seq2SeqIntentModel>` model.
- **examples/intent_extraction/train_mtl_model.py**: training script to train a :py:class:`MultiTaskIntentModel <nlp_architect.models.intent_extraction.MultiTaskIntentModel>` model.
- **examples/intent_extraction/interactive.py**: Inference script to run an input sentence using a trained model.

Running Modalities
==================

Training
--------

An example for training a multi-task model (predicts slot tags and intent type) using SNIPS dataset:

.. code:: python

  python examples/intent_extraction/train_mtl_model.py --dataset_path <dataset path> -b 10 -e 10


An example for training an Encoder-Decoder model (predicts only slot tags) using SNIPS, GloVe word embedding model of size 100 and saving the model weights to `my_model.h5`:

.. code:: python

  python examples/intent_extraction/train_enc-dec_model.py \
    --embedding_model <path_to_glove_100_file> \
    --token_emb_size 100 \
    --dataset_path <path_to_data> \
    --model_path my_model.h5


To list all possible parameters: ``python train_joint_model.py/train_enc-dec_model.py -h``

Interactive mode
----------------

Interactive mode allows to run sentences on a trained model (either of two) and get the results of the models displayed interactively.
The interactive session requires the dataset that the model was trained with for parsing new sentences.
Example:

.. code:: python

  python examples/intent_extraction/interactive.py --model_path my_model.h5 --dataset_path <path_to_data>

Results
=======

Results for SNIPS NLU dataset and ATIS are published below. The reference results were taken from the originating paper.
Minor differences might occur in final results. Each model was trained for 100 epochs with default parameters.

**SNIPS**

.. csv-table::
  :header: " ",Joint task, Encoder-Decoder
  :widths: 20, 40, 40
  :escape: ~

  Slots,97,85.96
  Intent,99.14, " "

**ATIS**

.. csv-table::
  :header: " ", "Joint task", "Encoder-Decoder", "[1]", "[2]"
  :widths: 20, 40, 40, 20, 20
  :escape: ~

  Slots,95.52,93.74,95.48,95.47
  Intent,96.08, , ,

.. note::

  We used ATIS [3]_ [4]_ dataset from: https://github.com/Microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/Data. Intel does not grant any rights to the data files.

References
----------

.. [1] Hakkani-Tur, Dilek and Tur, Gokhan and Celikyilmaz, Asli and Chen, Yun-Nung and Gao, Jianfeng and Deng, Li and Wang, Ye-Yi. `Multi-Domain Joint Semantic Frame Parsing using Bi-directional RNN-LSTM <https://www.csie.ntu.edu.tw/~yvchen/doc/IS16_MultiJoint.pdf>`_.

.. [2] Gakuto Kurata, Bing Xiang, Bowen Zhou, Mo Yu. `Leveraging Sentence-level Information with Encoder LSTM for Semantic Slot Filling <https://arxiv.org/abs/1601.01530>`_.

.. [3] C. Hemphill, J. Godfrey, and G. Doddington, The TabularIntentDataset spoken language systems pilot corpus, in Proc. of the DARPA speech and natural language workshop, 1990.

.. [4] P. Price, Evaluation of spoken language systems: The TabularIntentDataset domain, in Proc. of the Third DARPA Speech and Natural Language Workshop. Morgan Kaufmann, 1990.

.. [5] Alice Coucke and Alaa Saade and Adrien Ball and Théodore Bluche and Alexandre Caulier and David Leroy and Clément Doumouro and Thibault Gisselbrecht and Francesco Caltagirone and Thibaut Lavril and Maël Primet and Joseph Dureau. `Snips Voice Platform: an embedded Spoken Language Understanding system for private-by-design voice interfaces. <https://arxiv.org/abs/1805.10190>`_

.. _https://github.com/snipsco/nlu-benchmark: https://github.com/snipsco/nlu-benchmark
.. _here: https://medium.com/snips-ai/benchmarking-natural-language-understanding-systems-google-facebook-microsoft-and-snips-2b8ddcf9fb19
.. _configure: https://keras.io/backend/
.. _https://github.com/snipsco/nlu-benchmark/blob/master/LICENSE: https://github.com/snipsco/nlu-benchmark/blob/master/LICENSE
.. _https://github.com/Microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/Data: https://github.com/Microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/Data
.. _https://github.com/Microsoft/CNTK/blob/master/LICENSE.md: https://github.com/Microsoft/CNTK/blob/master/LICENSE.md
