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
..    :header: "Module API", "Description", "User Guide"
..    :widths: 20, 40, 30
..    :delim: |
..
..    :py:mod:`nlp_architect.data` | Data loading and handling | :doc:`Data loading<loading_data>`, :doc:`Datasets<datasets>`
..    :py:mod:`nlp_architect.models` | Model architecture | :doc:`Models<models>`

``nlp_architect.models``
---------------
.. py:module:: nlp_architect.models

Model classes stores a list of layers describing the model. Methods are provided
to train the model weights, perform inference, and save/load the model.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   np2vec.np2vec.NP2vec
   chunker.model.ChunkerModel
   intent_extraction.model.JointSequentialLSTM
   intent_extraction.model.EncDecTaggerModel
   models.np_semantic_segmentation.model.NpSemanticSegClassifier
   models.bist.bmstparser.mstlstm.MSTParserLSTM
   memn2n_dialogue.model.MemN2N_Dialog
   kvmemn2n.model.KVMemN2N


``nlp_architect.layers``
---------------------------
.. py:module:: nlp_architect.layers

In addition to imported layers, the Intel AI Architect contains its own set of layers.
These are currently stored in the various models

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chunker.model.TimeDistributedRecurrentOutput
   chunker.model.TimeDistributedRecurrentLast
   chunker.model.TimeDistBiLSTM
   reading_comprehension.ngraph_implementation.layers.MatchLSTMCell_withAttention
   reading_comprehension.ngraph_implementation.layers.AnswerPointer_withAttention
   reading_comprehension.ngraph_implementation.layers.Dropout_Modified
   reading_comprehension.ngraph_implementation.layers.LookupTable
   memn2n_dialogue.model.ModifiedLookupTable



``nlp_architect.data``
---------------------------
.. py:module:: nlp_architect.data

Currently datasets are distributed among the various models. In future versions of the code
these will be placed into a central repository.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    chunker.data.TaggedTextSequence
    chunker.data.MultiSequenceDataIterator
    chunker.data.CONLL2000
    chunker.model.DataInput
    intent_extraction.data.IntentDataset
    intent_extraction.data.ATIS
    intent_extraction.data.SNIPS
    models.np_semantic_segmentation.data.NpSemanticSegData
    memn2n_dialogue.data.BABI_Dialog
    kvmemn2n.data.WIKIMOVIES
