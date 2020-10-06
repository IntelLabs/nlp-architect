#! /usr/bin/env bash
python3.6 -m pip install -U pip setuptools virtualenv
python3.6 -m venv libert_env
source libert_env/bin/activate
git clone --branch libert https://github.com/NervanaSystems/nlp-architect.git
pip install -U pip

pip install -r /home/daniel_nlp/nlp-architect/nlp_architect/models/libert/requirements.txt
python -m spacy download en_core_web_lg

export SE_URL=https://raw.githubusercontent.com/HKUST-KnowComp/RINANTE/master/rinante-data
export LAPTOPS_FILES=laptops_test_sents.json,laptops_test_texts_tok_pos.txt,laptops_train_sents.json,laptops_train_texts_tok_pos.txt
export RESTAURANTS_FILES=restaurants_test_sents.json,restaurants_test_texts_tok_pos.txt,restaurants_train_sents.json,restaurants_train_texts_tok_pos.txt
export KDD_URL=https://raw.githubusercontent.com/happywwy/Recursive-Neural-Structural-Correspondence-Network/master/util/data_semEval
export KDD_FILES=addsenti_device,aspect_op_device

mkdir -p nlp-architect/nlp_architect/models/libert/data/Dai2019/semeval14/laptops && cd $_
curl -# -L "$SE_URL/semeval14/laptops/{$LAPTOPS_FILES}" -O --remote-name-all
mkdir ../restaurants && cd $_
curl -# -L "$SE_URL/semeval14/restaurants/{$RESTAURANTS_FILES}" -O --remote-name-all
mkdir -p ../../semeval15/restaurants && cd $_
curl -# -L "$SE_URL/semeval15/restaurants/{$RESTAURANTS_FILES}" -O --remote-name-all
mkdir -p ../../../Wang2018 && cd $_
curl -# -L "$KDD_URL/{$KDD_FILES}" -O --remote-name-all
cd ../..

python preprocess.py
python add_linguistic_info.py
