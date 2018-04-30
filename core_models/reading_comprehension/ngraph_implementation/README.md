
# Machine Reading Comprehension using Match LSTM and Answer Pointer

This directory contains an implementation of the boundary model(b in the Figure) Match LSTM and Answer Pointer network for Machine Reading Comprehension in Intel-Ngraph. The idea behind this method is to build a question aware representation of the passage and use this representation as an input to the pointer network which identifies the start and end indices of the answer.

<b>Model Architecture</b>
![Model architecture](https://github.com/NervanaSystems/ai-lab-models/blob/sharath/SQUAD_ngraph/ReadingComprehension/ngraph_implementation/MatchLSTM_Model.png)

# Dependencies:
The primary dependency of this model is `ngraph-python`. Download and installation instructions can be found at https://github.com/NervanaSystems/ngraph-python/blob/master/legacy_README.md. In addition to this, other required libraries are
- `numpy`
- `python3`

# Files:
- `train.py` -Implements the end to end model along with the training commands
- `utils.py`- Implements different utility functions to set up the data loader and to do the evaluation.
- `layers.py` -Contains different layers in ngraph required for the model
- `weight_initializers.py` - Containsm functions to initialize the LSTM Cell in ngraph

## Dataset
1. Download the training and  from here
https://rajpurkar.github.io/SQuAD-explorer/
2. Preprocess the data set using this command- `python squad_preprocess.py`

## Training
Train the model using the following command
 `python train.py -bgpu --gpu_id 0`

The command line options available are:
- `--gpu_id` select the gpu id train the model. Default is 0.
- `--max_para_req` enter the max length of the para to truncate the dataset.Default is 100. Currently the code has been tested for a maximum length of paragraph length of 100.
- `--batch_size_squad` enter the batch size (please note that 30 is the max batch size that will fit on the gpu with 12 gb memory). Default is 16.

## Results
After training starts, you will see outputs like this:
```
Loading Embeddings
creating training Set
Train Set Size is 19260
Dev set size is 2000
compiling the graph
generating transformer
iteration = 1, train loss = 13.156427383422852
F1_Score and EM_score are 0.0 0.0
iteration = 21, train loss = 12.441322326660156
F1_Score and EM_score are 8.333333333333332 0.0
iteration = 41, train loss = 10.773386001586914
F1_Score and EM_score are 6.25 6.25
iteration = 61, train loss = 11.69123649597168
F1_Score and EM_score are 6.25 6.25
```
Please note that after each epoch you will see the validation F1 and EM scores being printed out. These numbers are a result of a much stricter evaluation and lower than the official evaluation numbers.

Considering the default setting, which has training set of 19260 examples and a development set of 2000 examples
after 15 epochs, you should expect to see a F1 and EM scores on the development set similar to this:

```
F1 Score ~35%
EM Score ~25%
```
