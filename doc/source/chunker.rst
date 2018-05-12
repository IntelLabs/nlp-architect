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

Sequence Chunker
#############

Overview
========

Phrase chunking is a basic NLP task that consists of tagging parts of a sentence (1 or more tokens)
syntactically, i.e. POS tagging.

.. code:: bash

	The quick brown fox jumped over the fence
	|                   |      |    |
	Noun                Verb   Prep Noun

In this example the sentence can be divided into 4 phrases, ``The quick brown fox`` and ``the fence``
are noun phrases, ``jumped`` is a verb phrase and ``over`` is a prepositional phrase.

Dataset
=======

We used the CONLL2000_ shared task dataset in our example for training a phrase chunker.

The dataset has a ``train`` and ``test`` sets consisting of 8926 and 2009 sentences annotated with Part-of-speech and chuking information.
We implemented a dataset loader, ``nlp_architect.data.conll2000.CONLL2000``, for loading and parsing CONLL2000 data into iterators ready to be used by the chunker model.
``CONLL2000`` supports the following feature generation when loading the dataset:

1. Sentence words as sparse int representation
2. Pre-train word embeddings
3. Part-of-speech tags of words
4. Word character sparse representation (for extracting character features)

Model
=====

The Chunker model example comes with several options for creating the NN topology depending on what
input is given (tokens/POS/embeddings/char features).

.. image :: assets/model_diag.png

The model above depicts the main topology.
Given sentence ``S`` of length ``n``, and sentence tokens ``S = (s1, s2, .. , sn)`` we can input
vectors ``x1, x2, .., xn`` to the model where each sentence position ``i`` is a vector consisting
of the following values:

* token vector embedding using pre-trained word embedding
* token vector embedding (trained by model)
* part-of-speech embedding (trained by model)
* character features vector (trained by char-rnn)

.. image:: assets/char_diag.png

The Char-RNN feature extractor model uses 2 layers of LSTM such that each RNN layer outputs the
last hidden state. The final feature vector for a token is a concatenation of final hidden state of
the forward layer ``Hf`` and the backward ``Hb``. In the above example, the word ``apple`` is encoded to vector ``[Hf|Hb]``.

Following input vectors are 2 layers of LSTM cells, one LSTM reads input sentence from the token at
index ``1`` to ``n`` and the other backwards from ``n`` until ``1``. At each time step the forward
LSTM layer's hidden state is concatenated with the backward LSTM hidden state, and then used in a MLP
that predicts the token's tag at position ``i`` using a softmax activation layer. Eventually, the
model output are the tokens tags ``(tag_1, tag_2, .., tagn)`` as predicted in each step.

Deep Bi-directional LSTM
------------------------

In addition to the model described above, the model support the use of multiple stacked LSTM layers
as recent literature has indicated that several layers of RNN layers might be beneficial int sequential prediction.
When using multiple BiLSTM layers the hidden state of the forward and backward layers are at step ``i``
are used as the input to the next layer of BiLSTM at step ``i`` accordingly.


Running Models
==============

Training
--------
Quick train
^^^^^^^^^^^
Train a model with default parameters (use sentence words and default network settings):

.. code:: python

	python train.py

Custom training parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^
All customizable parameters can be obtained by running: ``python train.py -h``

.. code:: bash

	  --use_w2v             Use pre-trained word embedding from given w2v model
                        path (default: False)
	  --w2v_path W2V_PATH   w2v embedding model path (only GloVe and Fasttext are
	                        supported (default: None)
	  --use_pos             Use part-of-speech tags of tokens (default: False)
	  --use_char_rnn        Use char-RNN features of tokens (default: False)
	  --sentence_len SENTENCE_LEN
	                        Sentence token length (default: 100)
	  --lstm_depth LSTM_DEPTH
	                        Deep BiLSTM depth (default: 1)
	  --lstm_hidden_size LSTM_HIDDEN_SIZE
	                        LSTM cell hidden vector size (default: 128)
	  --token_embedding_size TOKEN_EMBEDDING_SIZE
	                        Token embedding vector size (default: 50)
	  --pos_embedding_size POS_EMBEDDING_SIZE
	                        Part-of-speech embedding vector size (default: 25)
	  --vocab_size VOCAB_SIZE
	                        Vocabulary size to use (only if pre-trained embedding
	                        is not used) (default: 25000)
	  --char_hidden_size CHAR_HIDDEN_SIZE
	                        Char-RNN cell hidden vector size (default: 25)
	  --model_name MODEL_NAME
	                        Model file name (default: chunker)
	  --settings SETTINGS   Model settings file name (default: chunker_settings)
	  --print_np_perf       Print Noun Phrase (NP) tags accuracy (default: True)


The model will automatically save after training is complete:

* ``<chunker>`` - Neon NN model file
* ``<chunker>_settings.dat`` - Model topology and input settings

Inference
---------
Quick inference
^^^^^^^^^^^^^^^

Running inference on a trained model ``chunker`` and ``chunker_settings.dat`` on input samples from ``inference_sentences.txt``

.. code:: python

	python inference.py --model chunker --parameters chunker_settings.dat --input inference_sentences.txt

Run ``python inference.py -h`` for a full list of options:

.. code:: bash

	  --model MODEL         Path to model file (default: None)
	  --settings SETTINGS   Path to model settings file (default: None)
	  --input INPUT         Input texts file path (samples to pass for inference)
	                        (default: None)
	  --emb_model EMB_MODEL
	                        Pre-trained word embedding model file path (default:
	                        None)
	  --print_only_nps      Print inferred Noun Phrases (default: False)

.. note::
	currently char-RNN feature (character embedding) is not supported in inference mode (will be added in the future).

Evaluation
==========
The reported performance below is on Noun Phrase (NP) detection (using B-NP and consecutive I-NP labels).

.. csv-table::
    :header: "Model", "Precision", "Recall", "F1"
    :widths: 40, 20, 20, 20
    :escape: ~

		CRF, 0.964, 0.964, 0.964
		Our model, 0.985, 0.959, 0.971


.. _CONLL2000: https://www.clips.uantwerpen.be/conll2000/chunking/
