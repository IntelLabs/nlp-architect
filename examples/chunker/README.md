# Sequence Chunker

Phrase chunking is a basic NLP task that consists of tagging parts of a sentence (1 or more tokens) syntactically.  

## Example

```text
The quick brown fox jumped over the fence
|                   |      |    |
Noun                Verb   Prep Noun
```

In this example the sentence can be divided into 4 phrases, `The quick brown fox` and `the fence` are noun phrases, `jumped` is a verb phrase and `over` is a prepositional phrase.

## Documentation

This model is based on the paper: [Deep multi-task learning with low level tasks supervised at lower layers](http://anthology.aclweb.org/P16-2038). \
Full documentation of this example and the neural network model can be found here: [http://nlp_architect.nervanasys.com/chunker.html](http://nlp_architect.nervanasys.com/chunker.html)

## Dataset

We used CONLL2000 in our example for training a phrase chunker. More info about the CONLL2000 shared task can be found [here](https://www.clips.uantwerpen.be/conll2000/chunking/).
The terms and conditions of the data set license apply. Intel does not grant any rights to the data files. The annotation of the data has been derived from the WSJ corpus by a program written by Sabine Buchholz from Tilburg University, The Netherlands.

NLP Architect has a dataloader for CONLL2000 which can be found in `nlp_architect.data.sequential_tagging.CONLL2000`

To get the dataset follow these steps:

- download train and test files from dataset website.
- unzip files: `gunzip *.gz`
- provide `CONLL2000` dataloader or `train.py` sample below to the directory containing the files.



## Training

Train a model with default parameters (only tokens, default network settings):  

```bash
python train.py --data_dir <path to CONLL2000 files>
```

Saving the model after training is done automatically by specifying a model name with the keyword `--model_name`, the following files will be created:

* `chunker_model.h5` - model file
* `chunker_model.params` - model parameter files (topology parameters, vocabs)

## Inference

Running inference on a trained model using an input file (text based, each line is a document):

```python
python inference.py --model_name <model_name> --input <input_file>.txt
```
