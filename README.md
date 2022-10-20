# Cross-Domain Aspect Extraction using Transformers Augmented with Knowledge Graphs

Repository for aspect extraction with knowledge-enhanced transformers as described in our CIKM 2022 paper:

> [Cross-Domain Aspect Extraction using Transformers Augmented with Knowledge Graphs](http://arxiv.org/abs/2210.10144).
> Phillip Howard, Arden Ma, Vasudev Lal, Ana Paula Simoes, Daniel Korat, Oren Pereg, Moshe Wasserblat and Gadi Singer.
> Proceedings of the 31st ACM International Conference on Information & Knowledge Management (CIKM '22).

## Installation

Install the required python packages. For machines with apt:
```
apt-get update && apt-get install python3.8 python3.8-dev python3.8-venv python3.8-distutils
```

Clone the aspect_extraction_with_kg branch of NLP Architect:
```
git clone -b aspect_extraction_with_kg https://github.com/IntelLabs/nlp-architect.git
cd nlp-architect
```

Create a new virtual environment and update pip:
```
python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

For machines with CUDA 10:
```
pip install -r requirements.txt
```
For machines with CUDA 11:
```
pip install -r requirements.txt --extra-index-url=https://download.pytorch.org/whl/cu113
```

## Download model files

```
wget -O nlp_architect/models/aspect_extraction_with_kg/bert-base-uncased/config.json https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
wget -O nlp_architect/models/aspect_extraction_with_kg/bert-base-uncased/pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
```

## Prepare KGs and data (optional)

Our domain-specific KGs and datasets are included in this repository. If you would like to manually generate the KGs and datasets, first run ``prepare_kg.ipynb`` to prepare the domain-specific KGs. By default, this notebook is configured to query the public ConceptNet API, which is very slow. We strongly recommend that you setup your own local ConceptNet API server and then modify ``cn_url`` in ``prepare_kg.ipynb`` to point to your local API endpoint. For instructions on setting up a local ConceptNet API server, see [this page](https://www.cs.utah.edu/~tli/posts/2018/09/blog-post-3/).

The ``prepare_kg.ipynb`` notebook will produce a file named ``seed_dist.pkl`` which contains the domain-specific knowledge graphs. Once this file has been created, you can then run the ``inject_knowledge.ipynb`` notebook to inject knowledge into the datasets for training and inference. 

## Run experiments

Prior to running the experiments, you may want to modify the model configuration files based on the number of available GPUs. The configuration files are located in ``nlp_architect/models/aspect_extraction_with_kg/config/``. By default, the experiments will be split across 3 GPUs with device IDs speicfied by the ``gpus`` list of the config file.

To run the BERT-PT experiment:

```
cd nlp_architect/models/aspect_extraction_with_kg/
python run.py bert_pt
```

To run the DeBERTa-PT experiment:

```
cd nlp_architect/models/aspect_extraction_with_kg/
python run_ray.py deberta_pt
```

To run the DeBERTa-MA experiment:

```
cd nlp_architect/models/aspect_extraction_with_kg/
python run_ray.py deberta_ma
```

After running an experiment, the mean aspect F1 for each cross-domain setting will be printed. See ``nlp_architect/models/aspect_extraction_with_kg/logs/latest/results.csv`` for more detailed results.
