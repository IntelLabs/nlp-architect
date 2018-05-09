# End-to-End Memory Network for Goal Oriented Dialogue
This directory contains an implementation of an End-to-End Memory Network for goal oriented dialogue in ngraph. 

## Datasets 
The dataset used for training and evaluation is under the umbrella of the Facebook bAbI dialog tasks (https://research.fb.com/downloads/babi/). The dataset can be downloaded from the command line if not found, and the preprocessing all happens at the beginning of training.

## Training
To train the model without match type on full dialog tasks, and test on the test set at the end, the following command can be used:
`python train_model.py --task 5 --weights_save_path memn2n_weights.npz --test`

The flag `--use_match_type` can also be used to enable match type features (for improved out-of-vocab performance but slower training).

## Interactive 
To begin interactive evaluation with a trained model, the following command can be used:
`python interactive.py --task 5 --model_file memn2n_weights.npz`

