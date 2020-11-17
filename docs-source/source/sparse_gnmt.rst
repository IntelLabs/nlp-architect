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

Compression of Google Neural Machine Translation Model
########################################################

Overview
========
Google Neural Machine Translation (GNMT) is a Sequence to sequence (Seq2seq) model which learns a mapping from an input text to an output text. \

The example below demonstrates how to train a highly sparse GNMT model with minimal loss in accuracy. The model is based on the *GNMT model presented in the paper Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation* [1]_ which consists of approximately 210M floating point parameters.

GNMT Model
==========
The GNMT architecture is an encoder-decoder architecture with attention as presented in the original paper [1]_.

The encoder consists of an embedding layer followed by 1 bi-directional and 3 uni-directional LSTM layers with residual connections between them.
The decoder consists of an embedding layer followed by 4 uni-directional LSTM layers and a linear Softmax layer.
The attention mechanism connects between the encoder's bi-directional LSTM layer to all of the decoder's LSTM layers.

The GNMT model was adapted from the model shown in *Neural Machine Translation (seq2seq) Tutorial* [2]_ and from its repository_.

The Sparse model implementation can be found in :py:class:`GNMTModel <examples.sparse_gnmt.gnmt_model.GNMTModel>` and offers several options to build the GNMT model.

Sparsity - Pruning GNMT
=======================
Sparse neural networks are networks where a portion of the network weights are zeros.
A high sparsity ratio can help compress the model and accelerate inference, reduce power consumption used for memory transfer and computing.

In order to produce a sparse network the network weights are pruned while training by forcing weights to be zero.
There are a number of methods to prune neural networks, for example the paper *To prune, or not to prune: exploring the efficacy of pruning for model compression* [3]_ presents a method for gradual pruning of weights with low amplitude.

The example below demonstrates how to prune the GNMT model up to 90% sparsity with minimal loss in BLEU score using the Tensorflow model_pruning_ package which implements the method presented in [3]_

Post Training Weight Quantization
=================================
The weights of pre-trained GNMT models are usually represented in 32bit Floating-point format.
The highly sparse pre-trained model below can be further compressed by uniform quantization of the weights to 8bits Integer, gaining a further compression ratio of 4x with negligible accuracy loss.
The implementation of the weight quantization is based on `TensorFlow API`_.
When using the model for inference, the int8 weights of the sparse and quantized model are de-quantized back to fp32.

Dataset
=======
The models below were trained using the following datasets:

- Europarlv7 [4]_
- Common Crawl Corpus
- News Commentary 11
- Development and test sets

All datasets are provided by `WMT Shared Task: Machine Translation of News`_

You can use this script wmt16_en_de.sh_ to download and prepare the data for training and evaluating your model.

Results & Pre-Trained Models
=============================
The following table presents some of our experiments and results. We provide pre-trained checkpoints for a 90% sparse GNMT model and a similar 90% sparse but with 2x2 sparsity blocks pattern. See table below and our `Model Zoo`_.
You can use these models to `Run Inference using our Pre-Trained Models`_ and evaluate them.

+----------------------------+----------+------+---------------------+-----------+
| Model                      | Sparsity | BLEU | Non-Zero Parameters | Data Type |
+----------------------------+----------+------+---------------------+-----------+
| Baseline                   |    0%    | 29.9 |        ~210M        |  Float32  |
+----------------------------+----------+------+---------------------+-----------+
| `Sparse`_                  |    90%   | 28.4 |         ~22M        |  Float32  |
+----------------------------+----------+------+---------------------+-----------+
| `2x2 Block Sparse`_        |    90%   | 27.8 |         ~22M        |  Float32  |
+----------------------------+----------+------+---------------------+-----------+
| Quantized Sparse           |    90%   | 28.4 |         ~22M        |  Integer8 |
+----------------------------+----------+------+---------------------+-----------+
| Quantized 2x2 Block Sparse |    90%   | 27.6 |         ~22M        |  Integer8 |
+----------------------------+----------+------+---------------------+-----------+

1. The pruning is applied to the embedding, decoder projection layer and all LSTM layers in both the encoder and decoder.
2. BLEU score is measured using *newstest2015* test set provided by the `Shared Task`_.
3. The accuracy of the quantized model was measure when we converted the 8 bits weights back to floating point during inference.


Running Modalities
==================
Below are simple examples for training 90% sparse :py:class:`GNMTModel <examples.sparse_gnmt.gnmt_model.GNMTModel>` model, running inference using a pre-trained/trained model, quantizing a model to 8bit Integer and running inference using a quantized model. Before inference, the int8 weights of the sparse and quantized model are de-quantize back to fp32.

Training
--------
Train a German to English GNMT model with 90% sparsity using the WMT16 dataset:

.. code-block:: bash

    # Download the dataset
    wmt16_en_de.sh /tmp/wmt16_en_de

    # Go to examples directory
    cd <nlp_architect root>/examples

    # Train the sparse GNMT
    python -m sparse_gnmt.nmt \
        --src=de --tgt=en \
        --hparams_path=sparse_gnmt/standard_hparams/sparse_wmt16_gnmt_4_layer.json \
        --out_dir=<output directory> \
        --vocab_prefix=/tmp/wmt16_en_de/vocab.bpe.32000 \
        --train_prefix=/tmp/wmt16_en_de/train.tok.clean.bpe.32000 \
        --dev_prefix=/tmp/wmt16_en_de/newstest2013.tok.bpe.32000 \
        --test_prefix=/tmp/wmt16_en_de/newstest2015.tok.bpe.32000

- Train using GPUs by adding ``--num_gpus=<n>``
- Model configuration JSON files are found in ``examples/sparse_gnmt/standard_hparams`` directory.
- Sparsity policy can be re-configured by changing the parameters given in ``--pruning_hparams``. E.g. change ``target_policy=0.7`` in order to train 70% sparse GNMT.
- All pruning hyper parameters are listed in model_pruning_.

While training Tensorflow checkpoints, Tensorboard events, Hyper-Parameters used and log files will be saved in the output directory given.

Inference
---------
Run inference using a trained model:

.. code-block:: bash

    # Go to examples directory
    cd <nlp_architect root>/examples

    # Run Inference
    python -m sparse_gnmt.nmt \
    --src=de --tgt=en \
    --hparams_path=sparse_gnmt/standard_hparams/sparse_wmt16_gnmt_4_layer.json \
    --ckpt=<path to a trained checkpoint> \
    --vocab_prefix=/tmp/wmt16_en_de/vocab.bpe.32000 \
    --out_dir=<output directory> \
    --inference_input_file=<file with lines in the source language> \
    --inference_output_file=<target file to place translations>

- Measure performance and BLEU score against a reference file by adding ``--inference_ref_file=<reference file in the target language>``
- Inference using GPUs by adding ``--num_gpus=<n>``

Run Inference using our Pre-Trained Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Run inference using our pre-trained models:

.. code-block:: bash

    # Download pre-trained model zip file, e.g. gnmt_sparse.zip
    wget https://d2zs9tzlek599f.cloudfront.net/models/sparse_gnmt/gnmt_sparse.zip

    # Unzip checkpoint + vocabulary files
    unzip gnmt_sparse.zip -d /tmp/gnmt_sparse_checkpoint

    # Go to examples directory
    cd <nlp_architect root>/examples

    # Run Inference
    python -m sparse_gnmt.nmt \
        --src=de --tgt=en \
        --hparams_path=sparse_gnmt/standard_hparams/sparse_wmt16_gnmt_4_layer.json \
        --ckpt=/tmp/gnmt_sparse_checkpoint/gnmt_sparse.ckpt\
        --vocab_prefix=/tmp/gnmt_sparse_checkpoint/vocab.bpe.32000 \
        --out_dir=<output directory> \
        --inference_input_file=<file with lines in the source language> \
        --inference_output_file=<target file to place translations>

*Important Note: use the vocabulary files provided with the checkpoint when using our pre-trained models*

Quantized Inference
^^^^^^^^^^^^^^^^^^^
Add the following flags to the `Inference`_ command line in order to quantize the pre-trained models and run inference with the quantized models:

- ``--quantize_ckpt=true``: Produce a quantized checkpoint. Checkpoint will be saved in the output directory. Inference will run using the produced checkpoint.
- ``--from_quantized_ckpt=true``: Inference using an already quantized checkpoint

Custom Training/Inference Parameters
------------------------------------
All customizable parameters can be obtained by running: ``python -m nlp-architect.examples.sparse_gnmt.nmt -h``

  -h, --help            show this help message and exit
  --num_units NUM_UNITS
                        Network size.
  --num_layers NUM_LAYERS
                        Network depth.
  --num_encoder_layers NUM_ENCODER_LAYERS
                        Encoder depth, equal to num_layers if None.
  --num_decoder_layers NUM_DECODER_LAYERS
                        Decoder depth, equal to num_layers if None.
  --encoder_type
                        uni | bi | gnmt. For bi, we build num_encoder_layers/2
                        bi-directional layers. For gnmt, we build 1 bi-
                        directional layer, and (num_encoder_layers - 1) uni-
                        directional layers.
  --residual
                        Whether to add residual connections.
  --time_major
                        Whether to use time-major mode for dynamic RNN.
  --num_embeddings_partitions NUM_EMBEDDINGS_PARTITIONS
                        Number of partitions for embedding vars.
  --attention
                        luong | scaled_luong | bahdanau | normed_bahdanau or
                        set to "" for no attention
  --attention_architecture
                        standard | gnmt | gnmt_v2. standard: use top layer to
                        compute attention. gnmt: GNMT style of computing
                        attention, use previous bottom layer to compute
                        attention. gnmt_v2: similar to gnmt, but use current
                        bottom layer to compute attention.
  --output_attention
                        Only used in standard attention_architecture. Whether
                        use attention as the cell output at each timestep.
  --pass_hidden_state
                        Whether to pass encoder's hidden state to decoder when
                        using an attention based model.
  --optimizer
                        sgd | adam
  --learning_rate LEARNING_RATE
                        Learning rate. Adam: 0.001 | 0.0001
  --warmup_steps WARMUP_STEPS
                        How many steps we inverse-decay learning.
  --warmup_scheme
                        How to warmup learning rates. Options include: t2t:
                        Tensor2Tensor's way, start with lr 100 times smaller,
                        then exponentiate until the specified lr.
  --decay_scheme
                        How we decay learning rate. Options include: luong234:
                        after 2/3 num train steps, we start halving the
                        learning rate for 4 times before finishing. luong5:
                        after 1/2 num train steps, we start halving the
                        learning rate for 5 times before finishing. luong10:
                        after 1/2 num train steps, we start halving the
                        learning rate for 10 times before finishing.
  --num_train_steps NUM_TRAIN_STEPS
                        Num steps to train.
  --colocate_gradients_with_ops
                        Whether try colocating gradients with corresponding op
  --init_op
                        uniform | glorot_normal | glorot_uniform
  --init_weight INIT_WEIGHT
                        for uniform init_op, initialize weights between
                        .
  --src SRC             Source suffix, e.g., en.
  --tgt TGT             Target suffix, e.g., de.
  --train_prefix TRAIN_PREFIX
                        Train prefix, expect files with src/tgt suffixes.
  --dev_prefix DEV_PREFIX
                        Dev prefix, expect files with src/tgt suffixes.
  --test_prefix TEST_PREFIX
                        Test prefix, expect files with src/tgt suffixes.
  --out_dir OUT_DIR     Store log/model files.
  --vocab_prefix VOCAB_PREFIX
                        Vocab prefix, expect files with src/tgt suffixes.
  --embed_prefix EMBED_PREFIX
                        Pretrained embedding prefix, expect files with src/tgt
                        suffixes. The embedding files should be Glove formatted
                        txt files.
  --sos SOS             Start-of-sentence symbol.
  --eos EOS             End-of-sentence symbol.
  --share_vocab
                        Whether to use the source vocab and embeddings for
                        both source and target.
  --check_special_token CHECK_SPECIAL_TOKEN
                        Whether check special sos, eos, unk tokens exist in
                        the vocab files.
  --src_max_len SRC_MAX_LEN
                        Max length of src sequences during training.
  --tgt_max_len TGT_MAX_LEN
                        Max length of tgt sequences during training.
  --src_max_len_infer SRC_MAX_LEN_INFER
                        Max length of src sequences during inference.
  --tgt_max_len_infer TGT_MAX_LEN_INFER
                        Max length of tgt sequences during inference. Also used
                        to restrict the maximum decoding length.
  --unit_type
                        lstm | gru | layer_norm_lstm | nas | mlstm
  --projection_type
                        dense | sparse
  --embedding_type
                        dense | sparse
  --forget_bias FORGET_BIAS
                        Forget bias for BasicLSTMCell.
  --dropout DROPOUT     Dropout rate (not keep_prob)
  --max_gradient_norm MAX_GRADIENT_NORM
                        Clip gradients to this norm.
  --batch_size BATCH_SIZE
                        Batch size.
  --steps_per_stats STEPS_PER_STATS
                        How many training steps to do per stats logging.Save
                        checkpoint every 10x steps_per_stats
  --max_train MAX_TRAIN
                        Limit on the size of training data (0: no limit).
  --num_buckets NUM_BUCKETS
                        Put data into similar-length buckets.
  --num_sampled_softmax NUM_SAMPLED_SOFTMAX
                        Use sampled_softmax_loss if > 0.Otherwise, use full
                        softmax loss.
  --subword_option
                        Set to bpe or spm to activate subword desegmentation.
  --use_char_encode USE_CHAR_ENCODE
                        Whether to split each word or bpe into character, and
                        then generate the word-level representation from the
                        character representation.
  --num_gpus NUM_GPUS   Number of gpus in each worker.
  --log_device_placement
                        Debug GPU allocation.
  --metrics METRICS     Comma-separated list of evaluations metrics
                        (bleu,rouge,accuracy)
  --steps_per_external_eval STEPS_PER_EXTERNAL_EVAL
                        How many training steps to do per external evaluation.
                        Automatically set based on data if None.
  --scope SCOPE         scope to put variables under
  --hparams_path HPARAMS_PATH
                        Path to standard hparams json file that
                        overrides hparams values from FLAGS.
  --random_seed RANDOM_SEED
                        Random seed (>0, set a specific seed).
  --override_loaded_hparams
                        Override loaded hparams with values specified
  --num_keep_ckpts NUM_KEEP_CKPTS
                        Max number of checkpoints to keep.
  --avg_ckpts
                        Average the last N checkpoints for external
                        evaluation. N can be controlled by setting
                        --num_keep_ckpts.
  --language_model
                        True to train a language model, ignoring encoder
  --ckpt CKPT           Checkpoint file to load a model for inference.
  --quantize_ckpt QUANTIZE_CKPT
                        Set to True to produce a quantized checkpoint from
                        existing checkpoint
  --from_quantized_ckpt FROM_QUANTIZED_CKPT
                        Set to True when the given checkpoint is quantized
  --inference_input_file INFERENCE_INPUT_FILE
                        Set to the text to decode.
  --inference_list INFERENCE_LIST
                        A comma-separated list of sentence indices (0-based)
                        to decode.
  --infer_batch_size INFER_BATCH_SIZE
                        Batch size for inference mode.
  --inference_output_file INFERENCE_OUTPUT_FILE
                        Output file to store decoding results.
  --inference_ref_file INFERENCE_REF_FILE
                        Reference file to compute evaluation scores (if
                        provided).
  --infer_mode
                        Which type of decoder to use during inference.
  --beam_width BEAM_WIDTH
                        beam width when using beam search decoder. If 0
                        (default), use standard decoder with greedy helper.
  --length_penalty_weight LENGTH_PENALTY_WEIGHT
                        Length penalty for beam search.
  --sampling_temperature SAMPLING_TEMPERATURE
                        Softmax sampling temperature for inference decoding,
                        0.0 means greedy decoding. This option is ignored when
                        using beam search.
  --num_translations_per_input NUM_TRANSLATIONS_PER_INPUT
                        Number of translations generated for each sentence.
                        This is only used for inference.
  --jobid JOBID         Task id of the worker.
  --num_workers NUM_WORKERS
                        Number of workers (inference only).
  --num_inter_threads NUM_INTER_THREADS
                        number of inter_op_parallelism_threads
  --num_intra_threads NUM_INTRA_THREADS
                        number of intra_op_parallelism_threads
  --pruning_hparams PRUNING_HPARAMS
                        model pruning parameters


References
==========
.. [1] Wu, Yonghui and Schuster, Mike and Chen, Zhifeng and Le, Quoc V and Norouzi, Mohammad and Macherey, Wolfgang and Krikun, Maxim and Cao, Yuan and Gao, Qin and Macherey, Klaus and others. Google's neural machine translation system: Bridging the gap between human and machine translation. https://arxiv.org/pdf/1609.08144.pdf
.. [2] Minh-Thang Luong and Eugene Brevdo and Rui Zhao. Neural Machine Translation (seq2seq) Tutorial. https://github.com/tensorflow/nmt
.. [3] Zhu, Michael and Gupta, Suyog. To prune, or not to prune: exploring the efficacy of pruning for model compression. https://arxiv.org/pdf/1710.01878.pdf
.. [4] A Parallel Corpus for Statistical Machine Translation, Philipp Koehn, MT Summit 2005

.. _repository: https://github.com/tensorflow/nmt
.. _model_pruning: https://github.com/tensorflow/tensorflow/tree/r1.10/tensorflow/contrib/model_pruning
.. _wmt16_en_de.sh: https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh
.. _`WMT Shared Task: Machine Translation of News`: http://www.statmt.org/wmt16/translation-task.html
.. _`Shared Task`: http://www.statmt.org/wmt16/translation-task.html
.. _`Model Zoo`: https://intellabs.github.io/nlp-architect/model_zoo.html
.. _`TensorFlow API`: https://www.tensorflow.org/api_docs/python/tf/quantize
.. _`Sparse`: https://d2zs9tzlek599f.cloudfront.net/models/sparse_gnmt/gnmt_sparse.zip
.. _`2x2 Block Sparse`: https://d2zs9tzlek599f.cloudfront.net/models/sparse_gnmt/gnmt_blocksparse2x2.zip
