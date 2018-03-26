# End-to-End Memory Network for Goal Oriented Dialogue
This directory contains an implementation of an End-to-End Memory Network for goal oriented dialogue in ngraph. 

Goal oriented dialogue is a subset of open-domain dialogue where an automated agent has a specific goal for the outcome of the interaction. At a high level, the system needs to understand a user request and complete a related task with a clear goal within a limited number of dialog turns. This task could be making a restaurant reservation, placing an order, setting a timer, or many of the digital personal assistant tasks.

End-to-End Memory Networks are generic semi-recurrent neural networks which allow for a bank of external memories to be read from and used during execution. They can be used in place of traditional slot-filling algorithms to accomplish goal oriented dialogue tasks without the need for expensive hand-labeled dialogue data. End-to-End Memory Netowrks have also been shown to be useful for Question-Answering and information retrieval tasks. 

## Dependencies: 
The dependencies required for the project are mentioned in requirements.txt. 
Use ```pip install -r requirements.txt```

<b>End-to-End Memory Network</b>
![E2EMemN2N Pic](https://camo.githubusercontent.com/ba1c7dbbccc5dd51d4a76cc6ef849bca65a9bf4d/687474703a2f2f692e696d6775722e636f6d2f6e7638394a4c632e706e67)

<b>Goal Oriented Dialog </b>
![Goal oriented dialog picture](https://i.imgur.com/5pQJqjM.png)

# Datasets 
The dataset used for training and evaluation is under the umbrella of the Facebook bAbI dialog tasks (https://research.fb.com/downloads/babi/). There are six separate tasks, tasks 1 through 5 are from simulated conversations between a customer and a restaurant booking bot (created by Facebook), and task 6 is more realistic natural language restaurant booking conversations as part of the dialog state tracking challenge (https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/).
=======
## Files

- <b>train_model.py</b>: Training script to load dataset and train memory network.
- <b>model.py</b>: Implementation of MemN2N_Dialog class for dialogue tasks. 
- <b>interactive.py</b>: Inference script to run interactive session with a trained goal oriented dialog agent.
- <b>utils.py</b>: Utilities to support interactive mode and simulate backend database.
- <b>data.py</b>: Data loader class to download data if not present and perform preprocessing.
- <b>tests.py</b>: Unit tests for custom lookuptable layer.

## Datasets 
The dataset used for training and evaluation is under the umbrella of the Facebook bAbI dialog tasks (https://research.fb.com/downloads/babi/). The dataset is automatically downloaded if not found, and the preprocessing all happens at the beginning of training.

There are six separate tasks, tasks 1 through 5 are from simulated conversations between a customer and a restaurant booking bot (created by Facebook), and task 6 is more realistic natural language restaurant booking conversations as part of the dialog state tracking challenge (https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/).

The descriptions of the six tasks are as follow:

- bAbI dialog dataset:
    - Task 1: Issuing API Calls
    - Task 2: Updating API Calls
    - Task 3: Displaying Options
    - Task 4: Providing Extra Information
    - Task 5: Conducting Full Dialogs

- Dialog State Tracking Challenge 2 Dataset:
    - Task 6: DSTC2 Full Dialogs

## Training
To train the model without match type on full dialog tasks, and test on the test set at the end, the following command can be used:
`python train_model.py --task 5 --weights_save_path memn2n_weights.npz --test`

The flag `--use_match_type` can also be used to enable match type features (for improved out-of-vocab performance but slower training).

## Interactive 
To begin interactive evaluation with a trained model, the following command can be used:
`python interactive.py --task 5 --model_file memn2n_weights.npz`

Interactive evaluation begins at the end of training and works as an interactive shell. Commands available for the shell are as follows:

- help: Display this help menu
- exit / quit: Exit interactive mode
- restart / clear: Restart the conversation and erase the bot's memory
- vocab: Display usable vocabulary
- allow_oov: Allow out of vocab words to be replaced with <OOV> token
- show_memory: Display the current contents of the bot's memory
- show_attention: Show the bot's memory & associated computed attention for the last memory hop

Otherwise, the interactive mode operates as a chat bot, responding to dialog to assist with restaurant booking. Vocabulary of the model is limited, please use the vocab command to see what the model actually understands.

## Results
The model was trained and evaluated on the 6 bAbI Dialog tasks with the following results.

| Task | This  | Published |  This (w/ match-type) | Published (w/ match-type)|
|------|--------|-----------| ---------------------|--------------------------|
| 1    | 99.8   | 99.9      | 100.0                | 100.0                    |
| 2    | 100.0  | 100.0     | 100.0                | 98.3                     |
| 3    | 74.8   | 74.9      | 74.6                 | 74.9                     |
| 4    | 57.2   | 59.5      | 100.0                | 100.0                    |
| 5    | 96.4   | 96.1      | 95.6                 | 93.4                     |
| 6    | 48.1   | 41.1      | 45.4                 | 41.0                     |

## Citations
<b>Paper</b>: A. Bordes, Y. Boureau, J. Weston. [Learning End-to-End Goal-Oriented Dialog](https://arxiv.org/abs/1605.07683). 2016<br>
<b>Reference TF Implementation</b>: [chatbot-MemN2N-tensorflow](https://github.com/vyraun/chatbot-MemN2N-tensorflow) (no match-type or interactive mode)
