# ABSA Solution

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
