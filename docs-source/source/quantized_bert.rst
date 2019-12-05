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

==============================================
Quantize BERT with Quantization Aware Training
==============================================

Overview
========
BERT - Bidirectional Encoder Representations from Transformers, is a language
representation model introduced last year by Devlin et al [1]_ .
It was shown that by fine-tuning a pre-trained BERT model it is possible to
achieve state-of-the-art performance on a wide variety of Natural Language
Processing (NLP) applications. \

In this page we are going to show how to run quantization aware training in the
fine tuning phase to a specific task in order to produce a quantized BERT
model which simulates quantized inference. In order to utilize
quantization for compressing the model's memory footprint or for
accelarating computation, true quantization must be applied using
optimized kernels and dedicated hardware.

.. note::
    :py:class:`QuantizedBertModel <nlp_architect.models.transformers.quantized_bert.QuantizedBertModel>` class does not make use of optimized Integer kernels for quantized Matrix Multiplication. Therefore, we expect that quantized inference will be slower than regular inference since quantization and dequantization operations are added to the unoptimized compute graph.

Quantization Aware Training
===========================
The idea of quantization aware training is to introduce to the model the
error caused by quantization while training in order for the model to learn
to overcome this error. \

In this work we use the quantization scheme and method offered by Jacob et
al [2]_. At the forward pass we use fake quantization to simulate the
quantization error during the forward pass and at the backward pass we estimate
the fake quantization gradients using Straight-Through Estimator [3]_.

Results
=======
The following table presents our experiments results. In the Quantization
Aware Training column we present the relative loss of accuracy w.r.t BERT
fine tuned to the specific task. Each result here is an average of 5
experiments. We used BERT-Base architecture and pre-trained model in all
the experiments except experiments with *-large* suffix which use the
BERT-Large architecture and pre-trained model.

+-------------+-----------------------+------------------------+---------------------+--------------------------------+--------------------+
|             | Metric                | BERT baseline accuracy | Quantized BERT 8bit | Relative Reduction of Accuracy | Dataset Size (*1k) |
+-------------+-----------------------+------------------------+---------------------+--------------------------------+--------------------+
| CoLA*       | Matthew's corr. (mcc) | 58.48                  | 58.48               | 0.00%                          | 8.5                |
+-------------+-----------------------+------------------------+---------------------+--------------------------------+--------------------+
| MRPC        | F1                    | 90                     | 89.56               | 0.49%                          | 3.5                |
+-------------+-----------------------+------------------------+---------------------+--------------------------------+--------------------+
| MRPC-Large  | F1                    | 90.86                  | 90.9                | -0.04%                         | 3.5                |
+-------------+-----------------------+------------------------+---------------------+--------------------------------+--------------------+
| QNLI        | Accuracy              | 90.3                   | 90.62               | -0.35%                         | 108                |
+-------------+-----------------------+------------------------+---------------------+--------------------------------+--------------------+
| QNLI-Large  | Accuracy              | 91.66                  | 91.74               | -0.09%                         | 108                |
+-------------+-----------------------+------------------------+---------------------+--------------------------------+--------------------+
| QQP         | F1                    | 87.84                  | 87.96               | -0.14%                         | 363                |
+-------------+-----------------------+------------------------+---------------------+--------------------------------+--------------------+
| RTE*        | Accuracy              | 69.7                   | 68.78               | 1.32%                          | 2.5                |
+-------------+-----------------------+------------------------+---------------------+--------------------------------+--------------------+
| SST-2       | Accuracy              | 92.36                  | 92.24               | 0.13%                          | 67                 |
+-------------+-----------------------+------------------------+---------------------+--------------------------------+--------------------+
| STS-B       | Pearson corr.         | 89.62                  | 89.04               | 0.65%                          | 5.7                |
+-------------+-----------------------+------------------------+---------------------+--------------------------------+--------------------+
| STS-B-Large | Pearson corr.         | 90.34                  | 90.12               | 0.24%                          | 5.7                |
+-------------+-----------------------+------------------------+---------------------+--------------------------------+--------------------+
| SQuAD       | F1                    | 88.46                  | 87.74               | 0.81%                          | 87                 |
+-------------+-----------------------+------------------------+---------------------+--------------------------------+--------------------+

Running Modalities
==================
In the following instructions for training and inference we use the `Microsoft
Research Paraphrase Corpus (MRPC)`_ which is included in the `GLUE benchmark`_
as an example dataset.

Training
--------
To train Quantized BERT use the following code snippet:

.. code-block:: bash

    nlp-train transformer_glue \
        --task_name mrpc \
        --model_name_or_path bert-base-uncased \
        --model_type quant_bert \
        --learning_rate 2e-5 \
        --output_dir /tmp/mrpc-8bit \
        --evaluate_during_training \
        --data_dir /path/to/MRPC \
        --do_lower_case

The model is saved at the end of training in 2 files:
   1. A model saved in FP32 for further ``pytorch_model.bin``
   2. A quantized model for inference only ``quant_pytorch_model.bin``

Inference
---------
To run inference with a fine tuned quantized BERT use the
following code snippet:

.. code-block:: bash

    nlp-inference transformer_glue \
        --model_path /tmp/mrpc-8bit \
        --task_name mrpc \
        --model_type quant_bert \
        --output_dir /tmp/mrpc-8bit \
        --data_dir /path/to/MRPC \
        --do_lower_case \
        --overwrite_output_dir

- To run evaluation on the task's development set add the flag ``--evaluate``
  to the command line.
- To run the quantized model saved in ``quant_pytorch_model.bin`` add the flag
  ``--load_quantized_model`` to the command line.


References
==========
.. [1] Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, https://arxiv.org/pdf/1810.04805.pdf
.. [2] Benoit Jacob and Skirmantas Kligys and Bo Chen and  Menglong Zhu and Matthew Tang and Andrew Howard and Hartwig Adam and Dmitry Kalenichenko, Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, https://arxiv.org/pdf/1712.05877.pdf
.. [3] Yoshua Bengio and Nicholas Leonard and Aaron Courville, Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation, https://arxiv.org/pdf/1308.3432.pdf

.. _`Microsoft Research Paraphrase Corpus (MRPC)`: https://www.microsoft.com/en-us/download/details.aspx?id=52398
.. _`GLUE benchmark`: https://gluebenchmark.com/
