# Set Expansion Solution

Term set expansion is the task of expanding a given partial set of terms into
a more complete set of terms that belong to the same semantic class. This
solution demonstrates the capability of a corpus-based set expansion system
in a simple web application.

## Flow

The solution is constructed of the following stages:

* Training:

* Loading the expand server with the trained model:
    ```
    python expand_server.py [--host HOST] [--port PORT] model_path
    ```
    The expand server gets requests containing seed terms, and expands them
    based on the given word embedding model. Important note: default server
    will listen on localhost:1234. If you set the host/port you should also
    set it in the ui/settings.py file.
* Run the UI application:
    ```
    bokeh serve --show ui
    ```
    The UI is a simple web based application for performing expansion.
    The application communicates with the expand server by sending expand
    requests, present the results in a simple table and export them to a csv
    file. Important note: If you set the host/port of the expand server you
    should also set it in the ui/settings.py file.

