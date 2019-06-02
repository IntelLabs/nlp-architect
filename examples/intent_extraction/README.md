# Intent Extraction


## Overview
Intent extraction is a type of Natural-Language-Understanding (NLU) task that helps to understand
the type of action conveyed in the sentences and all its participating parts.

An example of a sentence with intent could be:

```
Siri, can you please remind me to pickup my laundry on my way home?
```

The action conveyed in the sentence is to *remind* the speaker about something. The verb *remind*
applies that there is an assignee that has to do the action (who?), an assignee that the action
applies to (to whom?) and the object of the action (what?). In this case, *Siri* has to remind the
*speaker* to *pickup the laundry*.

## SNIPS NLU benchmark
A NLU benchmark containing ~16K sentences with 7 intent types. Each intent has about 2000 sentences

The dataset can be downloaded from [https://github.com/snipsco/nlu-benchmark](https://github.com/snipsco/nlu-benchmark). The terms and conditions of the data set license apply. Intel does not grant any rights to the data files.

Once the dataset is downloaded, point `<SNIPS folder/2017-06-custom-intent-engines` as the dataset path to `nlp_architect.data.intent_datasets.SNIPS`.

## Training
A quick example for training the multi-task model (predicts slot tags and intent type) using SNIPS dataset and saving the model weights to local file:

```
python examples/intent_extraction/train_mtl_model.py --dataset_path <dataset path> -b 10 -e 10
```

An example for training an Encoder-Decoder model (predicts slot tags) using SNIPS, GloVe word embedding model of size 100 and saving the model weights to `my_model.h5`:

```
python examples/intent_extraction/train_enc-dec_model.py \
    --embedding_model <path_to_glove_100_file> \
    --token_emb_size 100 \
    --dataset_path <path_to_data> \
    --model_path my_model.h5
```

to list all possible parameters using `-h` keyword.

## Interactive mode
Interactive mode allows to run sentences on a trained model (either of two) and get the results of the models displayed interactively.
Example:
```
python examples/intent_extraction/interactive.py --model_path model.h5 --model_info_path model_info.dat
```
