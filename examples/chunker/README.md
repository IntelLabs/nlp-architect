# BiLSTM Sequence Chunker

Phrase chunking is a basic NLP task that consists of tagging parts of a sentence (1 or more tokens) syntactically.  

### Example

	The quick brown fox jumped over the fence
	|                   |      |    |
	Noun                Verb   Prep Noun

In this example the sentence can be divided into 4 phrases, `The quick brown fox` and `the fence` are noun phrases, `jumped` is a verb phrase and `over` is a prepositional phrase.

## Documentation

Full documentation of this example and the neural network model can be found here: [http://nlp_architect.nervanasys.com/chunker.html](http://localhost:8000/chunker.html)

## Dataset

We used the CONLL2000 dataset in our example for training a phrase chunker. More info about the CONLL2000 shared task can be found [here](https://www.clips.uantwerpen.be/conll2000/chunking/).

If CONLL2000 is not found in NLTK, `nlp_architect.data.conll2000.CONLL2000` will attempt to download it by `nltk.download('conll2000')` after user consent.

The dataset can be downloaded from here: [https://www.nltk.org/data.html](https://www.nltk.org/data.html) The terms and conditions of the data set license apply. Intel does not grant any rights to the data files.

### Training
Train a model with default parameters (only tokens, default network settings):  

```
python train.py
```

Saving the model after training is done automatically:

* `<chunker>` - Neon NN model file
* `<chunker>_settings.dat` - Model topology and input settings

### Inference
Running inference on a trained model `chunker` and `chunker_settings.dat` on input samples from `inference_sentences.txt`

Quick example:
```
python inference.py --model chunker.prm --settings chunker_settings.dat --input inference_samples.txt
```  

**Note:** currently char-RNN features are not supported in inference models (will be added soon).
