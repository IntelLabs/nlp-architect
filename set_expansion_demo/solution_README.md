# Set Expansion Solution

Term set expansion is the task of expanding a given partial set of terms into
a more complete set of terms that belong to the same semantic class. This
solution demonstrates the capability of a corpus-based set expansion system
in a simple web application.


## Flow

The solution is constructed of the following stages:

1. Training:

   The first step in training is to prepare the data for generating a word embedding model.
   this is done by running:
   ```
    python3 mark_corpus.py -corpus train.txt -marked_corpus marked_train.txt
    ```
    the next step is to generate the model using NLP Architect np2vec module:
     ```
    python3 examples/np2vec/train.py --workers 30 --size 100 --min_count 10 --window 10 --hs 0 --corpus
    set_expansion_demo/marked_train.txt --np2vec_model_file set_expansion_demo/np2vec --corpus_format txt
  ```

2. Loading the expand server with the trained model:
    ```
    python expand_server.py [--host HOST] [--port PORT] model_path
    ```
    The expand server gets requests containing seed terms, and expands them
    based on the given word embedding model. You can use the model you trained
    yourself in the previous step, or to provide a pre-trained model you own.
    Important note: default server
    will listen on localhost:1234. If you set the host/port you should also
    set it in the ui/settings.py file.

3. Run the UI application:
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
