# Installation

```bash
python3.6 -m pip install -U pip setuptools virtualenv
python3.6 -m venv my_env
source my_env/bin/activate
git clone https://github.com/NervanaSystems/nlp-architect.git
pip install -e nlp-architect

git clone https://github.com/huggingface/transformers
cd transformers
pip install .
pip install -r ./examples/requirements.txt

pip install -U git+http://github.com/PyTorchLightning/pytorch-lightning/

```
