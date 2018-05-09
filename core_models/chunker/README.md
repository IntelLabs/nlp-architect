# BiLSTM Phrase Chunker

Phrase chunking is a basic NLP task that consists of tagging parts of a sentence (1 or more tokens) syntactically.  

### Example

  The quick brown fox jumped over the fence
  |                   |      |    |
  Noun                Verb   Prep Noun

In this example the sentence can be divided into 4 phrases, `The quick brown fox` and `the fence` are noun phrases, `jumped` is a verb phrase and `over` is a prepositional phrase.

## Dataset

We used the CoNLL2000 dataset in our example for training a phrase chunker. More info about this dataset can be found [here](https://www.clips.uantwerpen.be/conll2000/chunking/).

## Usage
### Training
Train a model with default parameters (only tokens, default network settings):  
	`python train.py`
Saving the model after training is done automatically:

* `<chunker>.prm` - Neon NN model file
* `<chunker>_settings.dat` - Model topology and input settings

### Inference
To run inference on a trained model one has to have a pre-trained chunker.prm and chunker_settings.dat model files. 

Quick example:
```
python inference.py --model chunker.prm --parameters chunker_settings.dat --input inference_samples.txt
```  

**Note:** currently char-RNN features are not supported in inference models (will be added soon).
