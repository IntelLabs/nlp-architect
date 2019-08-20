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

    nlp_architect.models.bist_parser.BISTModel
    nlp_architect.models.chunker.SequenceChunker
    nlp_architect.models.intent_extraction.Seq2SeqIntentModel
    nlp_architect.models.intent_extraction.MultiTaskIntentModel
    nlp_architect.models.matchlstm_ansptr.MatchLSTMAnswerPointer
    nlp_architect.models.memn2n_dialogue.MemN2N_Dialog
    nlp_architect.models.most_common_word_sense.MostCommonWordSense
    nlp_architect.models.ner_crf.NERCRF
    nlp_architect.models.np2vec.NP2vec
    nlp_architect.models.np_semantic_segmentation.NpSemanticSegClassifier
    nlp_architect.models.temporal_convolutional_network.TCN
    nlp_architect.models.crossling_emb.WordTranslator
    nlp_architect.models.cross_doc_sieves
    nlp_architect.models.cross_doc_coref.sieves_config.EventSievesConfiguration
    nlp_architect.models.cross_doc_coref.sieves_config.EntitySievesConfiguration
    nlp_architect.models.cross_doc_coref.sieves_resource.SievesResources
    nlp_architect.models.gnmt_model.GNMTModel


``nlp_architect.data``
----------------------
.. currentmodule:: nlp_architect.data

Dataset implementations and data loaders (check deep learning framework compatibility of dataset/loader in documentation)

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlp_architect.data.amazon_reviews.Amazon_Reviews
    nlp_architect.data.babi_dialog.BABI_Dialog
    nlp_architect.data.conll.ConllEntry
    nlp_architect.data.intent_datasets.IntentDataset
    nlp_architect.data.intent_datasets.TabularIntentDataset
    nlp_architect.data.intent_datasets.SNIPS
    nlp_architect.data.ptb.PTBDataLoader
    nlp_architect.data.sequential_tagging.CONLL2000
    nlp_architect.data.sequential_tagging.SequentialTaggingDataset
    nlp_architect.data.fasttext_emb.FastTextEmb
    nlp_architect.data.cdc_resources.relations.computed_relation_extraction.ComputedRelationExtraction
    nlp_architect.data.cdc_resources.relations.referent_dict_relation_extraction.ReferentDictRelationExtraction
    nlp_architect.data.cdc_resources.relations.verbocean_relation_extraction.VerboceanRelationExtraction
    nlp_architect.data.cdc_resources.relations.wikipedia_relation_extraction.WikipediaRelationExtraction
    nlp_architect.data.cdc_resources.relations.within_doc_coref_extraction.WithinDocCoref
    nlp_architect.data.cdc_resources.relations.word_embedding_relation_extraction.WordEmbeddingRelationExtraction
    nlp_architect.data.cdc_resources.relations.wordnet_relation_extraction.WordnetRelationExtraction
    nlp_architect.data.cdc_resources.relations.relation_types_enums.RelationType


``nlp_architect.pipelines``
---------------------------
.. currentmodule:: nlp_architect.pipelines

NLP pipelines modules using NLP Architect models

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlp_architect.pipelines.spacy_bist.SpacyBISTParser
    nlp_architect.pipelines.spacy_np_annotator.NPAnnotator
    nlp_architect.pipelines.spacy_np_annotator.SpacyNPAnnotator


``nlp_architect.contrib``
-------------------------
.. currentmodule:: nlp_architect.contrib

In addition to imported layers, the library also contains its own set of network layers and additions.
These are currently stored in the various models or related to which DL frameworks it was based on.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlp_architect.contrib.tensorflow.python.keras.layers.crf.CRF
    nlp_architect.contrib.tensorflow.python.keras.utils.layer_utils.save_model
    nlp_architect.contrib.tensorflow.python.keras.utils.layer_utils.load_model
    nlp_architect.contrib.tensorflow.python.keras.callbacks.ConllCallback


``nlp_architect.common``
------------------------
.. currentmodule:: nlp_architect.common

Common types of data structures used by NLP models

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlp_architect.common.core_nlp_doc.CoreNLPDoc
    nlp_architect.common.high_level_doc.HighLevelDoc
    nlp_architect.common.cdc.mention_data.MentionDataLight
    nlp_architect.common.cdc.mention_data.MentionData
