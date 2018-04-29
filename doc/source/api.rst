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

API
###

This API documentation covers each model within the Intel AI NLP Architect. Most modules have a
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
---------------
.. py:module:: models

Model classes stores a list of layers describing the model. Methods are provided
to train the model weights, perform inference, and save/load the model.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   nlp_architect.chunker.ChunkerModel
   nlp_architect.models.intent_extraction.JointSequentialLSTM
   nlp_architect.models.intent_extraction.EncDecTaggerModel
   nlp_architect.np2vec.np2vec.NP2vec
   nlp_architect.np_semantic_segmentation.model.NpSemanticSegClassifier
   nlp_architect.bist.bmstparser.mstlstm.MSTParserLSTM
   models.memn2n_dialogue.model.MemN2N_Dialog
   models.kvmemn2n.model.KVMemN2N


``nlp_architect.layers``
---------------------------
.. py:module:: nlp_architect.layers

In addition to imported layers, the Intel AI Architect contains its own set of layers.
These are currently stored in the various models

.. autosummary::
   :toctree: generated/
   :nosignatures:

   nlp_architect.chunker.model.TimeDistributedRecurrentOutput
   nlp_architect.chunker.model.TimeDistributedRecurrentLast
   nlp_architect.chunker.model.TimeDistBiLSTM
   models.reading_comprehension.ngraph_implementation.layers.MatchLSTMCell_withAttention
   models.reading_comprehension.ngraph_implementation.layers.AnswerPointer_withAttention
   models.reading_comprehension.ngraph_implementation.layers.Dropout_Modified
   models.reading_comprehension.ngraph_implementation.layers.LookupTable
   models.memn2n_dialogue.model.ModifiedLookupTable



``nlp_architect.data``
---------------------------
.. py:module:: nlp_architect.data

Currently datasets are distributed among the various models. In future versions of the code
these will be placed into a central repository.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlp_architect.chunker.data.TaggedTextSequence
    nlp_architect.chunker.data.MultiSequenceDataIterator
    nlp_architect.chunker.data.CONLL2000
    nlp_architect.chunker.model.DataInput
    nlp_architect.intent_extraction.data.IntentDataset
    nlp_architect.data.intent_datasets.ATIS
    nlp_architect.data.intent_datasets.SNIPS
    nlp_architect.np_semantic_segmentation.data.NpSemanticSegData
    models.memn2n_dialogue.data.BABI_Dialog
    models.kvmemn2n.data.WIKIMOVIES
