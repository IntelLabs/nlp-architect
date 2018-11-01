# Compression of Google Neural Machine Translation Model

## Overview
Google Neural Machine Translation (GNMT) is a Sequance to sequance (Seq2seq) model which learns a mapping from an input text to an output text. In the following example we use the GNMT model presented in the paper [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144) 
which consists ~210M floating point parameters. We demonstrate how to train a highly sparse or block-sparse model (90% sparse) with minimal accuracy loss. We also show how to further compress the highly sparse models by uniform quantization of the weights to 8bits Integer, gaining a further compression ratio of 4x with negligible accuracy loss. Before inference, we de-quantize the compressed int8 wights back to fp32.

## GNMT
The model in the example below is based on the [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt). Visit the repository for more information on the model and training process.

## Sparsity
In order to create a sparse model we prune the model weights while training i.e. we force some of the weights of the model to zero while training. \
In this example we make use of the Tensorflow [model_pruning](https://github.com/tensorflow/tensorflow/tree/r1.10/tensorflow/contrib/model_pruning) package which
implements the pruning method presented in the paper [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878)

## Dataset
In this example we use the following datasets:
* Europarlv7: A Parallel Corpus for Statistical Machine Translation, Philipp Koehn, MT Summit
2005
* Common Crawl Corpus
* News Commentary 11
* Development and test sets

All datasets are provided by [WMT Shared Task: Machine Translation of News](http://www.statmt.org/wmt16/translation-task.html)

You can use the script [wmt16_en_de.sh](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh) to download and prepare the data for training and evaluating

## Results & Pre-Trained Models
The following table presents some of our experiments and results.
Furthermore, some of our pre-trained models are offered in the form of checkpoints in our [Model Zoo](http://nlp_architect.nervanasys.com/model_zoo.html).
You can use these models to [Run Inference with Pre-Trained Model](#run-inference-with-pre-trained-model) and evaluate them.

| Model                      | Sparsity | BLEU| Non-Zero Parameters | Data Type |
|----------------------------|:--------:|:----:|:-------------------:|:---------:|
| Baseline                   |    0%    | 29.9 |        ~210M        |  Float32  |
| [Sparse](http://nervana-modelzoo.s3.amazonaws.com/NLP/gnmt/gnmt_sparse.zip)                     |    90%   | 28.4 |         ~22M        |  Float32  |
| [2x2 Block Sparse](http://nervana-modelzoo.s3.amazonaws.com/NLP/gnmt/gnmt_blocksparse2x2.zip)           |    90%   | 27.8 |         ~22M        |  Float32  |
| Quantized Sparse           |    90%   | 28.4 |         ~22M        |  Integer8 |
| Quantized 2x2 Block Sparse |    90%   | 27.6 |         ~22M        |  Integer8 |

1. The pruning is applied to embedding, decoder’s projection layer and all LSTM layers in both the encoder and decoder.
2. BLEU score is measured using *newstest2015* test set provided by the [Shared Task](http://www.statmt.org/wmt16/translation-task.html).
3. The accuracy of the quantized model was measure when we converted the 8 bits weights back to floating point during inference.    

## Training
Train 90% sparse GNMT model to translate from German to English

```    
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
```

* Train using GPUs by adding `--num_gpus=<n>`

## Inference
Running inference on a trained model

```

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
```

* Measure performance and BLEU score against a reference file by adding `--inference_ref_file=<reference file in the target language>`
* Inference using GPUs by adding `--num_gpus=<n>`

### Run Inference with Pre-Trained Model
Follow these instructions in order to use our pre-trained models:

```

# Download pre-trained model zip file, e.g. gnmt_sparse.zip
wget http://nervana-modelzoo.s3.amazonaws.com/NLP/gnmt/gnmt_sparse.zip

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
```

*Important Note: use the vocabulary files provided with the checkpoint when using our pre-trained models*

## Post Training Weight Quantization
Add these flags to the [Inference](#inference) command line in order to quantize model's weights and run quantized inference.

* `--quantize_ckpt=true`: Produce a quantized checkpoint. Checkpoint will be saved in the output directory. Inference will be done using the produced checkpoint
* `--from_quantized_ckpt=true`: Inference using an already quantized checkpoint
