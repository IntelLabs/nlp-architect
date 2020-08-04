# ABSApp

## Aspect-based Sentiment Analysis Solution

**Paper:** [ABSApp: A Portable Weakly-Supervised Aspect-Based Sentiment Extraction System](https://www.aclweb.org/anthology/D19-3001/)

### Abstract

We present ABSApp, a portable system for weakly-supervised aspect-based sentiment ex- traction. The system is interpretable and user friendly and does not require labeled training data, hence can be rapidly and cost-effectively used across different domains in applied setups. The system flow includes three stages: First, it generates domain-specific aspect and opinion lexicons based on an unlabeled dataset; second, it enables the user to view and edit those lexicons (weak supervision); and finally, it enables the user to select an unlabeled target dataset from the same domain, classify it, and generate an aspect-based sentiment report. ABSApp has been successfully used in a number of real-life use cases, among them movie review analysis and convention impact analysis.

## Video Demo

[![Video Demo](https://raw.githubusercontent.com/NervanaSystems/nlp-architect/absa/nlp_architect/solutions/absa_solution/assets/demo_screenshot.png)](https://drive.google.com/open?id=1BLk0xkjIOqyRhNy4UQEFQpDF_KR_NMAd)


## Setup

```bash
python3.6 -m pip install -U pip setuptools virtualenv
python3.6 -m venv absa_env
source absa_env/bin/activate
git clone --branch absa https://github.com/NervanaSystems/nlp-architect.git
cd nlp-architect
pip install --upgrade pip
pip install -e .
pip install -r nlp_architect/solutions/absa_solution/requirements.txt
```

## Run - Served Locally

```bash
    python nlp_architect/solutions/absa_solution/ui.py
    open http://localhost:5006
```

## Run - Served Remotely

Replace REMOTE_HOST with the server's hostname, and USER with your username:

```bash
    ssh USER@REMOTE_HOST -L 5006:REMOTE_HOST:5006
    export BOKEH_ALLOW_WS_ORIGIN=127.0.0.1:5007
    python nlp_architect/solutions/absa_solution/ui.py
```

Go to:  
[http://localhost:5006](http://localhost:5006)

## Citation

```bibtex
@article{pereg2019absapp,
  title={ABSApp: A Portable Weakly-Supervised Aspect-Based Sentiment Extraction System},
  author={Pereg, Oren and Korat, Daniel and Wasserblat, Moshe and Mamou, Jonathan and Dagan, Ido},
  journal={EMNLP-IJCNLP 2019},
  pages={1},
  year={2019}
}
```
