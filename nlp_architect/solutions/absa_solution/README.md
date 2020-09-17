# ABSApp - Aspect Based Sentiment Analysis Solution 
## Part of NLP Architect by Intel® AI Lab

**Demo paper @ EMNLP 2019:** [ABSApp: A Portable Weakly-Supervised Aspect-Based Sentiment Extraction System](https://www.aclweb.org/anthology/D19-3001/)

## Overview

Aspect Based Sentiment Analysis (ABSA) is the task of co-extracting **opinion terms** and **aspect terms** (opinion targets) and the relations between them in a given corpus.
Producing sentiment knowledge at the aspect level (vs. sentence-level) provides gains in **targeted business insight**.

**ABSApp** is a portable system for **weakly-supervised** ABSA. The system is **interpretable** and user friendly and **does not require labeled training data**, hence can be rapidly and cost-effectively used across different domains in applied setups. The system flow includes three stages: First, it generates domain-specific aspect and opinion lexicons based on an unlabeled dataset; second, it enables the user to view and edit those lexicons (weak supervision); and finally, it enables the user to select an unlabeled target dataset from the same domain, classify it, and generate an aspect-based sentiment report. ABSApp has been successfully used in a number of real-life use cases, among them movie review analysis and convention impact analysis.

## Video Demo

[![Video Demo](https://raw.githubusercontent.com/NervanaSystems/nlp-architect/absa/nlp_architect/solutions/absa_solution/assets/video.png)](https://drive.google.com/open?id=1BLk0xkjIOqyRhNy4UQEFQpDF_KR_NMAd)
*Figure 1*


## Workflow and UI

![Workflow](http://nlp_architect.nervanasys.com/_images/absa_solution_workflow.png)  
*Figure 2*



![Workflow](https://raw.githubusercontent.com/NervanaSystems/nlp-architect/absa/nlp_architect/solutions/absa_solution/assets/absa_solution_ui_3.png)
*Figure 3*  



**The 3 steps of the solution are:**

**Step 1:** The first step of the flow is to select an input dataset for lexicon extraction, performed by clicking the ‘Extract lexicons’ button shown in Figure 3. Once a dataset<sup>1</sup> is selected, the system performs the lexicon extraction process. Note that, this step can be skipped, in case the user already has aspect and opinion lexicons. In order to load pre-trained aspect and opinion lexicons select `Edit Lexicons` -> `Load` (Figure 3). For demonstration purposes, we provide pre-trained lexicons that are located at `examples/aspects.csv` and `examples/opinion.csv`. We also provide, a sample dataset<sup>2</sup> for lexicon extraction, at `datasets/absa/tripadvisor_co_uk-travel_restaurant_reviews_sample_2000_train.csv`.

**Step 2:** The user can choose to edit an aspect lexicon or an opinion lexicon that were generated in the previous step by selecting the `Aspect Lexicon` or `Opinion Lexicon` tab (see Figure 3). As shown in Figure 3, in which the `Aspect Lexicon` has been selected, the `Term` column displays the aspect terms while the `Alias1-3` columns display aspect terms that have the same semantic meaning. Upon selecting a specific aspect, the `Examples` view on the right-hand side, displays text snippets from the input dataset that include this term (highlighted in blue). the user can delete (by unchecking the term’s checkbox), add or modify the lexicon items. The opinion lexicon editor (not shown) functions similarly to the aspect lexicon editor except that it includes a `Polarity` column and a `Score` column. Both the polarity and the score can be edited by the user.

**Step 3:** A target dataset<sup>1</sup> and its classification are performed by clicking the `Classify` button in Figure 3. Once the dataset is selected the system starts the sentiment classification process and generates visualization of the sentiment analysis results under the ‘Analysis’ tab (Figure 1). For demonstration purposes we provide a sample classification dataset<sup>2</sup>, located under `datasets/absa/tripadvisor_co_uk-travel_restaurant_reviews_sample_2000_test.csv`.

<sup>1</sup> The format of the input dataset to steps 1 and 3 is a single raw text file with documents separated by newlines or a single csv file containing one doc per line or a directory containing one raw text file per document or a directory that includes parsed text files.

<sup>2</sup> Restaurants reviews from tripadvisor.co.uk under the Creative Commons Attribution-Share-Alike 3.0 License (Copyright 2018 Wikimedia Foundation).


## Setup

- **Create virtual environment (optional)**:

```bash
python3.6 -m pip install -U pip setuptools virtualenv
python3.6 -m venv absa_env
source absa_env/bin/activate
```

- **Clone and install**:

```bash
git clone --branch absa https://github.com/NervanaSystems/nlp-architect.git
pip install -U pip
pip install -e nlp-architect
pip install -r nlp-architect/nlp_architect/solutions/absa_solution/requirements.txt
```

## Run

### Served locally

```bash
    python nlp-architect/nlp_architect/solutions/absa_solution/ui.py
    open http://localhost:5006
```

### Served remotely

Replace `USER@REMOTE_HOST` with your username and server's hostname.

```bash
    ssh USER@REMOTE_HOST -L 5006:REMOTE_HOST:5006
    export BOKEH_ALLOW_WS_ORIGIN=127.0.0.1:5007
    python nlp-architect/nlp_architect/solutions/absa_solution/ui.py
```

When running for the first time, close the remote session (`Ctrl-D`) and reconnect, to update the environment variable:

```bash
    ssh USER@REMOTE_HOST -L 5006:REMOTE_HOST:5006
    source absa_env/bin/activate
    python nlp-architect/nlp_architect/solutions/absa_solution/ui.py
```

Open web browser to:  
[http://localhost:5006](http://localhost:5006)

## Citation

```bibtex
@inproceedings{pereg2019absapp,
  title={ABSApp: A Portable Weakly-Supervised Aspect-Based Sentiment Extraction System},
  author={Pereg, Oren and Korat, Daniel and Wasserblat, Moshe and Mamou, Jonathan and Dagan, Ido},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP): System Demonstrations},
  pages={1--6},
  year={2019}
}
```
