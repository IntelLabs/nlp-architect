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

Language Modeling
#################


Overview
========

A language model (LM) is a probability distribution over a sequence of words. Given a sequence, a trained language model can provide the probability that the sequence is realistic. Using deep learning, one manner of creating an LM is by training a neural network to predict the probability of occurrence of the next word (or character) in the sequence given all the words (or characters) preceding it. (In other words, the joint distribution over elements in a sequence is broken up using the chain rule.)

This folder contains scripts that implement a word-level language model using Temporal Convolutional Network (TCN) as described in the paper `An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling <https://arxiv.org/abs/1803.01271>`_ by Shaojie Bai, J. Zico Kolter and Vladlen Koltun. In this paper, the authors show that TCNs architectures are competitive with RNNs  across a diverse set of discrete sequence tasks. For language modeling, it is shown that TCN's performance on two datasets (Penn Tree Bank and WikiText) is comparable to that of an optimized LSTM architecture (with recurrent and embedding dropout, etc).



Data Loading
============
- PTB can be downloaded from `here <http://www.fit.vutbr.cz/~imikolov/rnnlm/>`_

- Wikitext can be downloaded from `here <https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset>`_

- The terms and conditions of the data set licenses apply. Intel does not grant any rights to the data files or databases.

- For the language modeling task, dataloader for the Penn tree bank (:py:class:`PTB <nlp_architect.data.ptb.PTBDataLoader>`) dataset (or the Wikitext-103 dataset) can be imported as

    .. code-block:: python

        from nlp_architect.data.ptb import PTBDataLoader, PTBDictionary

- Note that the data loader prompts the user to automatically download the data if not already present. Please provide the location to save the data as an argument to the data loader.

Running Modalities
==================
Training
--------
The base class that defines :py:class:`TCN <nlp_architect.models.temporal_convolutional_network.TCN>` topology can be imported as:

.. code-block:: python

    from nlp_architect.models.temporal_convolutional_network import TCN


Note that this is only the base class which defines the architecture. For defining a full trainable model, inherit this class and define the methods `build_train_graph()`, which should define the loss functions, and `run()`, which should define the training method.

For the language model, loss functions and the training strategy are implemented in `examples/word_language_model_with_tcn/mle_language_model/language_modeling_with_tcn.py`.

To train the model using PTB, use the following command:

.. code-block:: python

  python examples/word_language_model_with_tcn/mle_language_model/language_modeling_with_tcn.py \
    --batch_size 16 --dropout 0.45 --epochs 100 --ksize 3 --levels 4 --seq_len 60 \
    --nhid 600 --em_len 600 --em_dropout 0.25 --lr 4 \
    --grad_clip_value 0.35 --results_dir ./ --dataset PTB

The following tensorboard snapshots shows the result of a training run; plots for the training loss, perplexity, validation loss and perplexity are provided. With :py:class:`TCN <nlp_architect.models.temporal_convolutional_network.TCN>, we get word perplexity of 97 on the py:class:`PTB <nlp_architect.data.ptb.PTBDataLoader>` dataset.

.. image:: assets/lm.png

Inference
---------
To run inference and generate sample data, run the following command:

.. code-block:: python

  python examples/word_language_model_with_tcn/mle_language_model/language_modeling_with_tcn.py \
    --dropout 0.45 --ksize 3 --levels 4 --seq_len 60 --nhid 600 --em_len 600 \
    --em_dropout 0.25 --ckpt <path to trained ckpt file> --inference --num_samples 100

Using the provided trained checkpoint file, this will generate and print samples to stdout.
Some sample "sentences" generated using the :py:class:`PTB <nlp_architect.data.ptb.PTBDataLoader>` are shown below:

::

    over a third hundred feet in control of u.s. marketing units and nearly three years ago as well
    as N N to N N has cleared the group for $ N and they 're the revenue of at least N decade a
    <unk> <unk> electrical electrical home home and pharmaceuticals was in its battle mr. <unk> said

    as <unk> by <unk> and young smoke could follow as a real goal of writers

    <unk> <unk> while <unk> fit with this plan to cut back costs

    about light trucks

    more uncertainty than recycled paper people

    new jersey stock exchanges say i mean a <unk> <unk> part of those affecting the <unk> or
    female <unk> reported an <unk> of photographs <unk> and national security pacific

    <unk> and ford had previously been an <unk> <unk> that is the <unk> taping of <unk>
    thousands in the <unk> of <unk> fuels

    <unk> and <unk> tv paintings

    book values of about N department stores in france
