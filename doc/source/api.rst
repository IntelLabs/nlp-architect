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

API
###

This API documentation covers each model within NLP Architect. Most modules have a
corresponding user guide section that introduces the main concepts. See this
API for specific function definitions.

.. .. csv-table::
..    :header: "Module API", "Description"
..    :widths: 20, 40
..    :delim: |
..
..    :py:mod:`nlp_architect.models` | Model architecture
..    :py:mod:`nlp_architect.layers` | Model layers
..    :py:mod:`nlp_architect.data` | Data loading and handling

``nlp_architect.models``
------------------------
.. currentmodule:: nlp_architect.models

Model classes stores a list of layers describing the model. Methods are provided
to train the model weights, perform inference, and save/load the model.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bist_parser.BISTModel
    chunker.SequenceChunker
    intent_extraction.Seq2SeqIntentModel
    intent_extraction.MultiTaskIntentModel
    kvmemn2n.KVMemN2N
    matchlstm_ansptr.MatchLSTM_AnswerPointer
    memn2n_dialogue.MemN2N_Dialog
    most_common_word_sense.MostCommonWordSense
    ner_crf.NERCRF
    np2vec.NP2vec
    np_semantic_segmentation.NpSemanticSegClassifier
    temporal_convolutional_network.TCN
    crossling_emb.WordTranslator
    gnmt_model.GNMTModel


``nlp_architect.data``
----------------------
.. currentmodule:: nlp_architect.data

Dataset implementations and data loaders (check deep learning framework compatibility of dataset/loader in documentation)

.. autosummary::
    :toctree: generated/
    :nosignatures:

    amazon_reviews.Amazon_Reviews
    babi_dialog.BABI_Dialog
    conll.ConllEntry
    intent_datasets.IntentDataset
    intent_datasets.TabularIntentDataset
    intent_datasets.SNIPS
    ptb.PTBDataLoader
    sequential_tagging.CONLL2000
    sequential_tagging.SequentialTaggingDataset
    wikimovies.WIKIMOVIES
    fasttext_emb.FastTextEmb


``nlp_architect.pipelines``
---------------------------
.. currentmodule:: nlp_architect.pipelines

NLP pipelines modules using models implemented from ``nlp_architect.models``.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    spacy_bist.SpacyBISTParser
    spacy_np_annotator.NPAnnotator
    spacy_np_annotator.SpacyNPAnnotator


``nlp_architect.contrib``
-------------------------
.. currentmodule:: nlp_architect.contrib

In addition to imported layers, the library also contains its own set of network layers and additions.
These are currently stored in the various models or related to which DL frameworks it was based on.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ngraph.modified_lookup_table.ModifiedLookupTable
    tensorflow.python.keras.layers.crf.CRF
    tensorflow.python.keras.utils.layer_utils.save_model
    tensorflow.python.keras.utils.layer_utils.load_model
    tensorflow.python.keras.callbacks.ConllCallback
