.. ---------------------------------------------------------------------------
.. Copyright 2017-2019 Intel Corporation
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

==============================
Transformer model distillation
==============================

Overview
========

Transformer models which were pre-trained on large corpora, such as BERT/XLNet/XLM, 
have shown to improve the accuracy of many NLP tasks. However, such models have two
distinct disadvantages - (1) model size and (2) speed, since such large models are 
computationally heavy.

One possible approach to overcome these cons is to use Knowledge Distillation (KD).
Using this approach a large model is trained on the data set and then used to *teach*
a much smaller and more efficient network. This is often referred to a Student-Teacher
training where a teacher network adds its error to the student's loss function, thus, 
helping the student network to converge to a better solution.

Knowledge Distillation
======================

One approach is similar to the method in Hinton 2015 [#]_. The loss function is
modified to include a measure of distributions divergence, which can be measured
using KL divergence or MSE between the logits of the student and the teacher network.

:math:`loss = w_s \cdot loss_{student} + w_d \cdot KL(logits_{student} / T || logits_{teacher} / T)`

where *T* is a value representing temperature for softening the logits prior to 
applying softmax. `loss_{student}` is the original loss of the student network 
obtained during regular training. Finally, the losses are weighted.

``TeacherStudentDistill``
-------------------------

This class can be added to support for distillation in a model.
To add support for distillation, the student model must include handling of training
using ``TeacherStudentDistill`` class, see ``nlp_architect.procedures.token_tagging.do_kd_training`` for 
an example how to train a neural tagger using a transformer model using distillation.

.. autoclass:: nlp_architect.nn.torch.distillation.TeacherStudentDistill
   :members:

Supported models
================

``NeuralTagger``
----------------

Useful for training taggers from Transformer models. :py:class:`NeuralTagger <nlp_architect.models.tagging.NeuralTagger>` model that uses LSTM and CNN based embedders are ~3M parameters in size (~30-100x smaller than BERT models) and ~10x faster on average.

Usage:

#. Train a transformer tagger using :py:class:`TransformerTokenClassifier <nlp_architect.models.transformers.TransformerTokenClassifier>` or using ``nlp-train transformer_token`` command
#. Train a neural tagger :py:class:`Neural Tagger <nlp_architect.models.tagging.NeuralTagger>` using the trained transformer model and use the :py:class:`TeacherStudentDistill <nlp_architect.nn.torch.distillation.TeacherStudentDistill>` model that was configured with the transformer model. This can be done using :py:class:`Neural Tagger <nlp_architect.models.tagging.NeuralTagger>`'s train loop or by using ``nlp-train tagger_kd`` command


.. note::
    More models supporting distillation will be added in next releases

Pseudo Labeling
================

This method can be used in order to produce pseudo-labels when training the student on unlabeled examples.
The pseudo-guess is produced by applying arg max on the logits of the teacher model, and results in the following loss:

.. math::

    loss &= \Bigg\{\begin{eqnarray}CE(yˆ, y) && labeled&example\\ CE(yˆ, yˆt) && unlabeled&example\end{eqnarray}


where CE is Cross Entropy loss, yˆ is the predicted entity label class by the student model and yˆt is
the predicted label by the teacher model.


.. [#] Distilling the Knowledge in a Neural Network: Geoffrey Hinton, Oriol Vinyals, Jeff Dean, https://arxiv.org/abs/1503.02531