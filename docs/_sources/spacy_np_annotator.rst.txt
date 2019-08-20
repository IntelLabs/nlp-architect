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

Spacy-NP Annotator
##################

Chunker based noun phrase annotator
===================================

The noun phrase annotator is a plug-in that can be used with Spacy_ pipeline structure.

The annotator loads a trained :py:class:`SequenceChunker <nlp_architect.models.chunker.SequenceChunker>` model that is able to predict chunk labels, creates Spacy based Span objects and applies a sequence of filtering to produce a set of noun phrases, finally, it attaches it to the document object.

The annotator implementation can be found in :py:class:`NPAnnotator <nlp_architect.pipelines.spacy_np_annotator.NPAnnotator>`.

Usage example
-------------
Loading a Spacy pipeline and adding a sentence breaker (required) and :py:class:`NPAnnotator <nlp_architect.pipelines.spacy_np_annotator.NPAnnotator>` annotator as the last annotator in the pipeline:

.. code:: python

    nlp = spacy.load('en')
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    nlp.add_pipe(NPAnnotator.load(<path_to_model>, <path_to_params>), last=True)

Parse documents regularly and get the noun phrase annotations using a dedicated method:

.. code:: python

    doc = nlp('The quick brown fox jumped over the fence')
    noun_phrases = nlp_architect.pipelines.spacy_np_annotator.get_noun_phrases(doc)


Standalone Spacy-NPAnnotator
============================

For use cases in which the user is not interested in specialized Spacy pipelines we have implemented :py:class:`SpacyNPAnnotator <nlp_architect.pipelines.spacy_np_annotator.SpacyNPAnnotator>` which will run a Spacy pipeline internally and provide string based noun phrase chunks given documents in string format.

Usage example
-------------

Just as in :py:class:`NPAnnotator <nlp_architect.pipelines.spacy_np_annotator.NPAnnotator>`, we need to provide a trained :py:class:`SequenceChunker <nlp_architect.models.chunker.SequenceChunker>` model and its parameters file. It is also possible to provide a specific Spacy model to base the pipeline on.

The following example shows how to load a model/parameters using the default Spacy English model (`en`) and how to get the noun phrase annotations.

.. code:: python

    spacy_np = SpacyNPAnnotator(<model_path>, <model_parameters_path>, spacy_mode='en')
    noun_phrases = spacy_np('The quick brown fox jumped over the fence')

.. _Spacy: https://spacy.io
