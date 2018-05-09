# BiLSTM Phrase Chunker
This directory contains an implementation of a BiLSTM Pharse Chunker which tags parts of sentence syntactically of a given input.

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
