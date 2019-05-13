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

NLP Architect Model Zoo
#######################

.. list-table::
   :widths: 10 30 10
   :header-rows: 1

   * - Model
     - Description
     - Links
   * - :doc:`Sparse GNMT <sparse_gnmt>`
     - 90% sparse GNMT model and a 2x2 block sparse translating German to English trained on Europarl-v7 [1]_ , Common Crawl and News Commentary 11 datasets
     -  | `model <https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/sparse_gnmt/gnmt_sparse.zip>`_
        | `2x2 block sparse model <https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/sparse_gnmt/gnmt_blocksparse2x2.zip>`_
   * - :doc:`Intent Extraction <intent>`
     - A :py:class:`MultiTaskIntentModel <nlp_architect.models.intent_extraction.MultiTaskIntentModel>` intent extraction and slot tagging model, trained on SNIPS NLU dataset
     - | `model <https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/intent/model.h5>`_
       | `params <https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/intent/model_info.dat>`_
   * - :doc:`Named Entity Recognition <ner_crf>`
     - A :py:class:`NERCRF <nlp_architect.models.ner_crf.NERCRF>` model trained on CoNLL 2003 dataset
     - | `model <https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/ner/model.h5>`_
       | `params <https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/ner/model_info.dat>`_
   * - :doc:`Dependency parser <bist_parser>`
     - Graph-based dependency parser using BiLSTM feature extractors
     - `model <https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/dep_parse/bist-pretrained.zip>`_
   * - :doc:`Machine comprehension <reading_comprehension>`
     - Match LSTM model trained on SQuAD dataset
     - | `model <https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/mrc/mrc_model.zip>`_
       | `data <https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/mrc/mrc_data.zip>`_
   * - :doc:`Word chunker <chunker>`
     - A word chunker model trained on CoNLL 2000 dataset
     - | `model <https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/chunker/model.h5>`_
       | `params <https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/chunker/model_info.dat.params>`_

References
==========
.. [1] Europarl-v7: A Parallel Corpus for Statistical Machine Translation, Philipp Koehn, MT Summit 2005