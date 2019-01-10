
# Machine Reading Comprehension using Match LSTM and Answer Pointer

This directory contains an implementation of the boundary model(b in the Figure) Match LSTM and Answer Pointer network for Machine Reading Comprehension in Tensorflow. The idea behind this method is to build a question aware representation of the passage and use this representation as an input to the pointer network which identifies the start and end indices of the answer.

## Dataset
1. mkdir data

2. Download the official SQuAD-v1.1 training (train-v1.1.json) and development(dev-v1.1.json) datasets  from here
https://worksheets.codalab.org/worksheets/0x62eefc3e64e04430a1a24785a9293fff/ and place the extracted json files in the `data` directory. For more information about SQuAD, please visit https://rajpurkar.github.io/SQuAD-explorer/. The terms and conditions of the data set license apply. Intel does not grant any rights to the data files.

3. Download the GloVe pretrained embeddings from  http://nlp.stanford.edu/data/glove.6B.zip and copy "glove.6B.300d.txt" file into the  `data` directory.
For more information about GloVe please visit https://nlp.stanford.edu/projects/glove/. The terms and conditions of the data set license apply. Intel does not grant any rights to the data files.

4. Preprocess the data set using this command- `python prepare_data.py --data_path data/`

## Training
Train the model using the following command
 `python train.py --data_path data/ --gpu_id 0`

## Inference
To visualize predicted answers for paragraphs and questions in the validation dataset use the following command

`python train.py --restore_model=True --inference_mode=True --data_path=data/ --model_dir=/path/to/trained_model/ --batch_size=1 --num_examples=50`

## Results
After training starts, you will see outputs as shown below:
```
Loading Embeddings
creating training and development sets
Match LSTM Pass
Answer Pointer Pass
Setting up Loss
Set up optimizer
Begin Training
Epoch Number:  0
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

Considering the default setting, which has training set of 85387 examples and a development set of 10130 examples
after 15 epochs, you should expect to see a F1 and EM scores on the development set similar to this:

```
F1 Score ~62%
EM Score ~48%
```

## Citations

1. SQuAD: 100,000+ Questions for Machine Comprehension of Text. Authors: Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang.
   Subjects: Computation and Language(cs.CL). arXiv:1606.05250 [cs.CL][https://arxiv.org/abs/1606.05250].
   License: https://creativecommons.org/licenses/by-sa/4.0/legalcode
2. Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014 https://nlp.stanford.edu/pubs/glove.pdf. License: http://www.opendatacommons.org/licenses/pddl/1.0/
3. Wang, S., & Jiang, J. (2016). Machine comprehension using match-lstm and answer pointer. arXiv preprint arXiv:1608.07905. [https://arxiv.org/abs/1608.07905]
