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

Named Entity Recognition
========================
``NeuralTagger``
----------------

A model for training token tagging tasks, such as NER or POS. ``NeuralTagger`` requires an **embedder** for 
extracting the contextual features of the data, see embedders below.
The model uses either a *Softmax* or a *Conditional Random Field* classifier to classify the words into 
correct labels. Implemented in PyTorch and support only PyTorch based embedders.

See :py:class:`NeuralTagger <nlp_architect.models.tagging.NeuralTagger>` for complete documentation of model methods.


.. autoclass:: nlp_architect.models.tagging.NeuralTagger

``CNNLSTM``
-----------

This module is a embedder based on `End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF`_ by Ma and Hovy (2016). 
The model uses CNNs to embed character representation of words in a sentence and stacked bi-direction LSTM layers to embed the context of words and characters.

.. figure:: ../assets/cnn-lstm-fig.png

   CNN-LSTM topology (taken from original paper)


**Usage**

Use :py:class:`TokenClsProcessor <nlp_architect.data.sequential_tagging.TokenClsProcessor>` for parsing input files for the model. :py:class:`NeuralTagger <nlp_architect.models.tagging.NeuralTagger>` for training/loading a trained model.

Training a model::

    nlp_architect train tagger --model_type cnn-lstm --data_dir <path to data dir> --output_dir <model output dir>

See ```nlp_architect train tagger -h``` for full list of options for training.

Running inference on trained model::

    nlp_architect run tagger --data_file <input data file> --model_dir <model dir> --output_dir <output dir>

See ```nlp_architect run tagger -h``` for full list of options for running a trained model.

.. autoclass:: nlp_architect.nn.torch.modules.embedders.CNNLSTM
   
``IDCNN``
---------


The module is an embedder based on `Fast and Accurate Entity Recognition with Iterated Dilated Convolutions`_ by Strubell et at (2017). 
The model uses Iterated-Dilated convolusions for sequence labelling. An dilated CNN block utilizes CNN and dilations to catpure the context of a whole sentence and relation ships between words.
In the figure below you can see an example for a dilated CNN block with maximum dilation of 4 and filter width of 3. 
This model is a fast alternative to LSTM-based models with ~10x speedup compared to LSTM-based models.

.. figure:: ../assets/idcnn-fig.png
   
   A dilated CNN block (taken from original paper)

We added a word character convolution feature extractor which is concatenated to the embedded word representations.

**Usage**

Use :py:class:`TokenClsProcessor <nlp_architect.data.sequential_tagging.TokenClsProcessor>` for parsing input files for the model. :py:class:`NeuralTagger <nlp_architect.models.tagging.NeuralTagger>` for training/loading a trained model.

Training a model::

    nlp_architect train tagger --model_type id-cnn --data_dir <path to data dir> --output_dir <model output dir>


See ```nlp_architect train tagger -h``` for full list of options for training.

Running inference on trained model::

    nlp_architect run tagger --data_file <input data file> --model_dir <model dir> --output_dir <output dir>


See ```nlp_architect run tagger -h``` for full list of options for running a trained model.

.. autoclass:: nlp_architect.nn.torch.modules.embedders.IDCNN

.. _transformer_cls:

``TransformerTokenClassifier``
------------------------------

A tagger using a Transformer-based topology and a pre-trained model on a large collection of data (usually wikipedia and such).

:py:class:`TransformerTokenClassifier <nlp_architect.models.transformers.TransformerTokenClassifier>` We provide token tagging classifier head module for Transformer-based pre-trained models. 
Currently we support BERT/XLNet and quantized BERT base models which utilize a fully-connected layer with *Softmax* classifier. Tokens which were broken into multiple sub-tokens (using Wordpiece algorithm or such) are ignored. For a complete list of transformer base models run ```nlp_architect train transformer_token -h``` to see a list of models that can be fine-tuned to your task. 

**Usage**

Use :py:class:`TokenClsProcessor <nlp_architect.data.sequential_tagging.TokenClsProcessor>` for parsing input files for the model. Depending on which model you choose, the padding and sentence formatting is adjusted to fit the base model you chose.

See model class :py:class:`TransformerTokenClassifier <nlp_architect.models.transformers.TransformerTokenClassifier>` for usage documentation.

Training a model::

    nlp_architect train transformer_token \
        --data_dir <path to data> \
        --model_name_or_path <name of pre-trained model or path> \
        --model_type [bert, quant_bert, xlnet] \
        --output_dir <path to output dir>

See ```nlp_architect train transformer_token -h``` for full list of options for training.

Running inference on a trained model::

    nlp_architect run transformer_token \
        --data_file <path to input file> \
        --model_path <path to trained model> \
        --model_type [bert, quant_bert, xlnet] \
        --output_dir <output path>

See ``nlp_architect run tagger -h`` for full list of options for running a trained model.

.. _BIO: https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
.. _`Lample et al.`: https://arxiv.org/abs/1603.01360
.. _`Neural Architectures for Named Entity Recognition`: https://arxiv.org/abs/1603.01360
.. _`Conditional Random Field classifier`: https://en.wikipedia.org/wiki/Conditional_random_field
.. _`End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF`: https://arxiv.org/abs/1603.01354
.. _`Fast and Accurate Entity Recognition with Iterated Dilated Convolutions`: https://arxiv.org/abs/1702.02098
.. _`Deep multi-task learning with low level tasks supervised at lower layers`: http://anthology.aclweb.org/P16-2038

