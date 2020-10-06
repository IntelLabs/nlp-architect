# LiBERT

## Requirements

- Linux (tested on Ubuntu)
- Python 3.6.x
- GPU + CUDA 10.1+

## Automated Setup

To set up code and data, execute the included script:


### For machines with CUDA 10.2:

```bash
curl -O  https://raw.githubusercontent.com/NervanaSystems/nlp-architect/libert/nlp_architect/models/libert/setup.sh
chmod +x setup.sh && ./setup.sh

```

### For machines with CUDA 10.1:

```bash
curl -O  https://raw.githubusercontent.com/NervanaSystems/nlp-architect/libert/nlp_architect/models/libert/setup-cu101.sh
chmod +x setup-cu101.sh && ./setup-cu101.sh

```

These scripts clone the code, install requirements, and download and prepare the data, as detailed in the following sections.

## Manual Setup

### Prepare environment and clone repo

```bash
python3.6 -m pip install -U pip setuptools virtualenv
python3.6 -m venv libert_env
source libert_env/bin/activate
git clone --branch libert https://github.com/NervanaSystems/nlp-architect.git
pip install -U pip
```

### Install requirements

#### For machines with CUDA 10.2:

```bash
pip install -r nlp-architect/nlp_architect/models/libert/requirements.txt
python -m spacy download en_core_web_lg
```

#### For machines with CUDA 10.1:

```bash
pip install -r nlp-architect/nlp_architect/models/libert/requirements_no_torch.txt
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
python -m spacy download en_core_web_lg
```

### Prepare Data

#### SemEval14-15 (Tokenized) Dataset and KDD-2004 Dataset

SemEval14-15 data contains reviews in restaurants and laptops domains. It can can be obtained from [this](https://github.com/HKUST-KnowComp/RINANTE) GitHub Repository ([MIT License](https://github.com/HKUST-KnowComp/RINANTE/blob/master/LICENSE)).
KDD-2004 data contains reviews the devices domains. It can can be obtained from [this](https://github.com/happywwy/Recursive-Neural-Structural-Correspondence-Network) GitHub Repository (unlicenesed).
Here are the commands for downloading and pre-processing train, dev and test datasets for laptops, restaurants and devices domains.

Let's define some variables for these steps:

```bash
export SE_URL=https://raw.githubusercontent.com/HKUST-KnowComp/RINANTE/master/rinante-data
export LAPTOPS_FILES=laptops_test_sents.json,laptops_test_texts_tok_pos.txt,laptops_train_sents.json,laptops_train_texts_tok_pos.txt
export RESTAURANTS_FILES=restaurants_test_sents.json,restaurants_test_texts_tok_pos.txt,restaurants_train_sents.json,restaurants_train_texts_tok_pos.txt
export KDD_URL=https://raw.githubusercontent.com/happywwy/Recursive-Neural-Structural-Correspondence-Network/master/util/data_semEval
export KDD_FILES=addsenti_device,aspect_op_device
```

Download the datasets into the required folder structure:

```bash
mkdir -p nlp-architect/nlp_architect/models/libert/data/Dai2019/semeval14/laptops && cd $_
curl -# -L "$SE_URL/semeval14/laptops/{$LAPTOPS_FILES}" -O --remote-name-all
mkdir ../restaurants && cd $_
curl -# -L "$SE_URL/semeval14/restaurants/{$RESTAURANTS_FILES}" -O --remote-name-all
mkdir -p ../../semeval15/restaurants && cd $_
curl -# -L "$SE_URL/semeval15/restaurants/{$RESTAURANTS_FILES}" -O --remote-name-all
mkdir -p ../../../Wang2018 && cd $_
curl -# -L "$KDD_URL/{$KDD_FILES}" -O --remote-name-all
cd ../..
```

Convert the datasets into CoNLL format, and then into CSV with added lingusitic info:

```bash
python preprocess.py
python add_linguistic_info.py
```

## Running experiments

Parameters for an experiment are defined in a YAML configuration file for that expermient.

- Create `CONFIG_NAME.yaml` file with hyperparams in config folder (see `example.yaml` file).

- Run:

    ```bash
    python run.py CONFIG_NAME
    ```

## Accessing TensorBoard remotely

- On remote machine:

    ```bash
    tensorboard --host REMOTE_HOST --port PORT --logdir path/to/nlp-architect/models/libert/logs
    ```

- On local machine:

    ```bash
    ssh user@REMOTE_HOST -L PORT:REMOTE_HOST:PORT
    ```

- Point browser to:

    `http://localhost:PORT/`
