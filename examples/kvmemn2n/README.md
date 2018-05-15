# End-to-End Memory Network for Goal Oriented Dialogue
This directory contains an implementation of an end-to-end key-value memory network for goal oriented dialog in ngraph.
The idea behind this method is to be able to answer wide range of questions based on a large set of textual information, as opposed to a restricted or sparse knowledge base.

# Dataset
Please download the tar file from http://www.thespermwhale.com/jaseweston/babi/movieqa.tar.gz and expand the folder into your desired data directory or `--data_dir`. The terms and conditions of the data set license apply. Intel does not grant any rights to the data files. The dataset can be downloaded from the command line if not found, and the preprocessing all happens at the beginning of training.

# Training
The base command to train is `python train_kvmemn2n.py`.
The following are example commands to run training using knowledge base and raw text respectively
```
python train_kvmemn2n.py --epochs 2000 --batch_size 32 --emb_size 100 --use_v_luts --model_file path_to_model_dir/kb_model
```
```
python train_kvmemn2n.py --mem_mode text --epochs 2000 --batch_size 32 --emb_size 50 --model_file path_to_model_dir/text_model
```

# Interactive Mode
You can enter an interactive mode using the argument `--interactive`. The interactive mode can be called to launch at the end of training, or direcly after `--inference`. To run inference on the KB model from above we would call:

```
python train_kvmemn2n.py --batch_size 32 --emb_size 100 --use_v_luts --model_file path_to_model_dir/kb_model --inference --interactive
```
Note that we set `--emb_size 100` and `--use_v_luts` as the original model used these parameters.
