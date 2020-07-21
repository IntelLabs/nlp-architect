# SA-BERT

## Installation

```bash
python3.6 -m pip install -U pip setuptools virtualenv
python3.6 -m venv my_env
source my_env/bin/activate
git clone https://github.com/NervanaSystems/nlp-architect.git
pip install -e nlp-architect
pip install -U pytorch transformers pytorch-lightning tensorboard scikit-learn seqeval numpy scipy
git branch sa-bert
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
    tensorboard --host REMOTE_HOST --port PORT --logdir path/to/nlp-architect/lightning_logs
    ```

- On local machine:

    ```bash
    ssh user@REMOTE_HOST -L PORT:REMOTE_HOST:PORT
    ```

- Point browser to:

    `http://localhost:PORT/`
