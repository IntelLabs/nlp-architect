# Set Expansion Solution

Term set expansion is the task of expanding a given partial set of terms into
a more complete set of terms that belong to the same semantic class. This
solution demonstrates the capability of a corpus-based set expansion system
in a simple web application.

![Image](assets/demo.png)

## Algorithm Overview
Our approach is based on representing any term of a training corpus using word embeddings in order 
to estimate the similarity between the seed terms and any candidate term. Noun phrases provide 
good approximation for candidate terms and are extracted in our system using a noun phrase chunker. 
At expansion time, given a seed of terms, the most similar terms are returned.

## Flow

The solution is constructed of the following stages:


## Training:
   
The first step in training is to prepare the data for generating a word embedding model.
This is done by running:
```
python prepare_data.py -corpus train.txt -marked_corpus marked_train.txt
```
The next step is to generate the model using [NLP Architect np2vec module](http://nlp_architect.nervanasys.com/np2vec.html):
```
python examples/np2vec/train.py --size 100 --min_count 10 --window 10 --hs 0 --corpus solutions/set_expansion/marked_train.txt --np2vec_model_file solutions/set_expansion/np2vec --corpus_format txt
```

## Inference:

It consists in expanding the seed terms. This can be done in two manners:

1. Running python script

2. Web application

2.1. Loading the expand server with the trained model:
    ```
    python expand_server.py [--host HOST] [--port PORT] model_path
    ```
    The expand server gets requests containing seed terms, and expands them
    based on the given word embedding model. You can use the model you trained
    yourself in the previous step, or to provide a pre-trained model you own.
    Important note: default server
    will listen on localhost:1234. If you set the host/port you should also
    set it in the ui/settings.py file.

2.2. Run the UI application:
    ```
    bokeh serve --show ui
    ```
    The UI is a simple web based application for performing expansion.
    The application communicates with the server by sending expand
    requests, present the results in a simple table and export them to a csv
    file. It allows you to either directly type the terms to expand or to
    select terms from the model vocabulary list. After you get some expand
    results you can perform re-expansion by selecting terms from the results (hold the Ctrl key for
    multiple selection).Important note: If you set the host/port of the expand server you
    should also set it in the ui/settings.py file. You can also load the ui
    application as a server using the bokeh options --address and --port, for example:
    ```
    bokeh serve ui --address=127.0.0.1 --port=1234 --allow-websocket-origin=127.0.0.1:1234
    ```

## Citation
[Term Set Expansion based on Multi-Context Term Embeddings: an End-to-end Workflow](https://drive.google.com/open?id=164MvUGo0-iPeuGM1b8XrH2ysZZFrzomF), Jonathan Mamou,
 Oren Pereg, Moshe Wasserblat, Ido Dagan, Yoav Goldberg, Alon Eirew, Yael Green, Shira Guskin, 
 Peter Izsak, Daniel Korat, COLING 2018.