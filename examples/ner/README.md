# Named Entity Recognition


## Overview
Named Entity Recognition (NER) is a basic Information extraction task in which words (or phrases) are classified into pre-defined entity groups (or marked as non interesting). Entity groups share common characteristics of consisting words or phrases and are identifiable by the shape of the word or context in which they appear in sentences. Examples of entity groups are: names, numbers, locations, currency, dates, company names, etc.

Example sentence:

```
	John is planning a visit to London on October
	|                           |         |
	Name                        City      Date
```

In this example, a `name`, `city` and `date` entities are identified.

## Quick train
Train a model with default parameters given input data files:

```
python train.py --train_file train.txt --test_file test.txt
```

## Interactive mode
Run a trained model in interactive mode and enter input from stdin.

```
python interactive.py --model_path model.h5 --model_info_path model_info.dat
```
