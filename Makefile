# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

STYLE_CHECK_OPTS :=
STYLE_CHECK_DIRS :=
DOC_DIR := doc
DOC_PUB_RELEASE_PATH := $(DOC_PUB_PATH)/$(RELEASE)
test_prepare:
	pip install -r test_requirements.txt > /dev/null 2>&1

style: test_prepare
	flake8 --output-file style.txt --tee $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS)
	pylint --reports=n --output-format=colorized --py3k $(PYLINT3K_ARGS) --ignore=.venv *

fixstyle: autopep8

autopep8:
	autopep8 -a -a --global-config setup.cfg --in-place `find . -name \*.py`
	echo run "git diff" to see what may need to be checked in and "make style" to see what work remains

doc_prepare:
	pip install -r doc_requirements.txt > /dev/null 2>&1

doc: doc_prepare
	$(MAKE) -C $(DOC_DIR) clean
	$(MAKE) -C $(DOC_DIR) html
	@echo "Documentation built in $(DOC_DIR)/build/html"
	@echo "To view documents open your browser to: http://localhost:8000"
	@cd $(DOC_DIR)/build/html; python -m http.server
	@echo

html:
	$(MAKE) -C $(DOC_DIR) html

LIBRARY_NAME := nlp_architect
VIRTUALENV_DIR := .nlp_architect_env
ACTIVATE := $(VIRTUALENV_DIR)/bin/activate
MODELS_DIR := $(LIBRARY_NAME)/models

$(ACTIVATE):
	pip3 install --upgrade pip setuptools virtualenv
	virtualenv -p python3 $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip install -U pip
	# @. $(basic); pip install -r requirements.txt

intent: $(ACTIVATE)
	@echo "installing intent extractor model"
	# @. $(ACTIVATE); pip install -r $(MODELS_DIR)/intent_extraction/requirements.txt
	# @$(MAKE) finally

chunker: $(ACTIVATE)
	@echo "installing chunker model"
	# @. $(ACTIVATE); pip install -r $(MODELS_DIR)/chunker/requirements.txt
	@$(MAKE) finally

bist: $(ACTIVATE)
	@echo "installing BIST parser"
	# @. $(ACTIVATE); pip install -r $(MODELS_DIR)/bist/requirements.txt
	@$(MAKE) finally

kbmemn2n: $(ACTIVATE)
	@echo "installing key-value memory network model"
	# @. $(ACTIVATE); pip install -r $(MODELS_DIR)/kvmemn2n/requirements.txt
	@$(MAKE) finally

mem2n_dialog: $(ACTIVATE)
	@echo "installing memory network for dialog model"
	# @. $(ACTIVATE); pip install -r $(MODELS_DIR)/memn2n_dialogue/requirements.txt
	@$(MAKE) finally

np_seg: $(ACTIVATE)
	@echo "installing NP semantic segmentation model"
	# @. $(ACTIVATE); pip install -r $(MODELS_DIR)/np_semantic_segmentation/requirements.txt
	@$(MAKE) finally

np2vec: $(ACTIVATE)
	@echo "installing NP2vec model"
	# @. $(ACTIVATE); pip install -r $(MODELS_DIR)/np2vec/requirements.txt
	@$(MAKE) finally

finally: $(ACTIVATE)
	@. $(ACTIVATE); pip install -e .
	@echo "\n\n****************************************"
	@echo "Setup complete."
	@echo "Type:"
	@echo "    . '$(ACTIVATE)'"
	@echo "to work interactively $(VIRTUALENV_DIR) virtual env"
