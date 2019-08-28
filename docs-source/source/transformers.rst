.. ---------------------------------------------------------------------------
.. Copyright 2016-2018 Intel Corporation
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, softw+are
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

============
Transformers
============

NLP Architect integrated the Transformer models available in `pytorch-transformers <https://github.com/huggingface/pytorch-transformers>`_. Using Transformer models based on a pre-trained models usually done by attaching a classification head on the transformer model and fine-tuning the model (transformer and classifier) on the target (down-stream) task.

Base model
----------

:py:class:`TransformerBase <nlp_architect.models.transformers.base_model.TransformerBase>` is a base class for handling 
loading, saving, training and inference of transformer models. 

The base model support `pytorch-transformers` configs, tokenizers and base models as documented in their `website <https://github.com/huggingface/pytorch-transformers>`_ (see our base-class for supported models).

In order to use the Transformer models just sub-class the base model and include:

* A classifier (head) for your task.
* sub-method handling of input to tensors used by model.
* any sub-method to evaluate the task, do inference, etc.

Models
------
Sequence classification
~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`TransformerSequenceClassifier <nlp_architect.models.transformers.sequence_classification.TransformerSequenceClassifier>` is a transformer model with sentence classification head (the ``[CLS]`` token is used as classification label) for sentence classification tasks (classification/regression). 

See ``nlp_architect.procedures.transformers.glue`` for an example of training sequence classification models on GLUE benchmark tasks.

Training a model on GLUE tasks, using BERT-base uncased base model:

.. code-block:: bash

    nlp_architect train transformer_glue \
        --task_name <task name> \
        --model_name_or_path bert-base-uncased \
        --model_type bert \
        --output_dir <output dir> \
        --evaluate_during_training \
        --data_dir </path/to/glue_task> \
        --do_lower_case

Running a model:

.. code-block:: bash

    nlp_architect run transformer_glue \
        --model_path <path to model> \
        --task_name <task_name> \
        --model_type bert \
        --output_dir <output dir> \
        --data_dir <path to data> \
        --do_lower_case \
        --overwrite_output_dir

Token classification
~~~~~~~~~~~~~~~~~~~~

:py:class:`TransformerTokenClassifier <nlp_architect.models.transformers.token_classification.TransformerTokenClassifier>` is a transformer model for token classification for tasks such as NER, POS or chunking.

See example for usage :ref:`transformer_cls` NER model description.




