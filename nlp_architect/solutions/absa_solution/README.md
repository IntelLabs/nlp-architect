# ABSApp

## Aspect Based Sentiment Analysis (ABSA) Solution

**Paper:** [ABSApp: A Portable Weakly-Supervised Aspect-Based Sentiment Extraction System](https://www.aclweb.org/anthology/D19-3001/)

### Overview

Aspect Based Sentiment Analysis is the task of co-extracting opinion terms and aspect terms (opinion targets) and the relations between them in a given corpus.

## Solution Overview

[![Video Demo](http://nlp_architect.nervanasys.com/_images/absa_solution_workflow.png)](https://drive.google.com/open?id=1BLk0xkjIOqyRhNy4UQEFQpDF_KR_NMAd)

The solution flow includes three stages: first, it generates domain-specific aspect and opinion lexicons based on an unlabeled dataset; second, it enables the user to view and edit those lexicons; and finally, it enables the user to select an unlabeled target dataset from the same domain, classify it, and generate an aspect-based sentiment report.

For lexicon extraction, the solution calls the training step of NLP Architect’s ABSA training, whereas for sentiment classification, the solution calls NLP Architect’s ABSA inference. For more details see ABSA.

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
