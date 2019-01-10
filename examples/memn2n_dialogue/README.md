# End-to-End Memory Network for Goal Oriented Dialogue
This directory contains an implementation of an End-to-End Memory Network for goal oriented dialogue in TensorFlow. 

Goal oriented dialogue is a subset of open-domain dialogue where an automated agent has a specific goal for the outcome of the interaction. At a high level, the system needs to understand a user request and complete a related task with a clear goal within a limited number of dialog turns. This task could be making a restaurant reservation, placing an order, setting a timer, or many of the digital personal assistant tasks.

End-to-End Memory Networks are generic semi-recurrent neural networks which allow for a bank of external memories to be read from and used during execution. They can be used in place of traditional slot-filling algorithms to accomplish goal oriented dialogue tasks without the need for expensive hand-labeled dialogue data. End-to-End Memory Networks have also been shown to be useful for Question-Answering and information retrieval tasks. 

## Datasets 
The dataset used for training and evaluation is under the umbrella of the Facebook bAbI dialog tasks (https://research.fb.com/downloads/babi/. License: https://github.com/facebook/bAbI-tasks/blob/master/LICENSE.md). The terms and conditions of the data set license apply. Intel does not grant any rights to the data files. The dataset can be downloaded from the command line if not found, and the preprocessing all happens at the beginning of training.

## Training
To train the model without match type on full dialog tasks, and test on the test set at the end, the following command can be used:
`python train_model.py --task 5 --weights_save_path saved_tf/ --test`

The flag `--use_match_type` can also be used to enable match type features (for improved out-of-vocab performance but slower training).

## Interactive 
To begin interactive evaluation with a trained model, the following command can be used:
`python interactive.py --task 5 --weight_save_path saved_tf/`

