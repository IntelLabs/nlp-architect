# LiBERT

## Installation

Intructions for machines with CIDA 10.2 installed:

```bash
python3.6 -m pip install -U pip setuptools virtualenv
python3.6 -m venv libert_env
source libert_env/bin/activate
git clone --branch libert https://github.com/NervanaSystems/nlp-architect.git
pip install -r -U nlp-architect/nlp_architect/models/libert/requirements.txt
```

## Usage

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
