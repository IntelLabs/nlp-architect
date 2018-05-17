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

FLAKE8_CHECK_DIRS := examples nlp_architect/* server tests
PYLINT_CHECK_DIRS := *
DOC_DIR := doc
DOC_PUB_RELEASE_PATH := $(DOC_PUB_PATH)/$(RELEASE)

LIBRARY_NAME := nlp_architect
VIRTUALENV_DIR := .nlp_architect_env
REQ_FILE := _generated_reqs.txt
ACTIVATE := $(VIRTUALENV_DIR)/bin/activate
MODELS_DIR := $(LIBRARY_NAME)/models
NLP_DIR := $(LIBRARY_NAME)/nlp

.PHONY: test_prepare test style doc_prepare doc html clean pre_install install dev install_no_virt_env sysinstall finish_install

default: dev

test_prepare: test_requirements.txt $(ACTIVATE)
	@. $(ACTIVATE); pip install -r test_requirements.txt > /dev/null 2>&1

test: test_prepare $(ACTIVATE) dev
	@. $(ACTIVATE); py.test -rs tests

style: test_prepare
	@. $(ACTIVATE); flake8 --exit-zero --output-file flake.txt --tee $(FLAKE8_CHECK_DIRS)
	@. $(ACTIVATE); pylint --reports=n --output-format=colorized --ignore=.venv $(PYLINT_CHECK_DIRS) || true

doc_prepare: doc_requirements.txt $(ACTIVATE)
	@. $(ACTIVATE); pip install -r doc_requirements.txt > /dev/null 2>&1

doc: doc_prepare
	@. $(ACTIVATE); $(MAKE) -C $(DOC_DIR) clean
	@. $(ACTIVATE); $(MAKE) -C $(DOC_DIR) html
	@echo "Documentation built in $(DOC_DIR)/build/html"
	@echo "To view documents open your browser to: http://localhost:8000"
	@. $(ACTIVATE); cd $(DOC_DIR)/build/html; python -m http.server
	@echo

html: doc_prepare $(ACTIVATE)
	@. $(ACTIVATE); $(MAKE) -C $(DOC_DIR) html

clean:
	@echo "Cleaning files.."
	@rm -rf $(VIRTUALENV_DIR)
	@rm -rf $(REQ_FILE)

ENV_EXIST := $(shell test -d $(VIRTUALENV_DIR) && echo -n yes)
REQ_EXIST := $(shell test -f $(REQ_FILE) && echo -n yes)

$(REQ_FILE):
ifneq ($(REQ_EXIST), yes)
	@echo "Generating pip requirements file"
	@bash generate_reqs.sh
endif

$(ACTIVATE): $(REQ_FILE)
ifneq ($(ENV_EXIST), yes)
	@echo "NLP Architect installation"
	@echo "**************************"
	@echo "Creating new environment"
	@echo
	virtualenv -p python3 $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip install -U pip
endif

pre_install: $(ACTIVATE)
	@echo "\n\n****************************************"
	@echo "Installing packages ..."
	@. $(ACTIVATE); pip install -r $(REQ_FILE)

install: pre_install
	@. $(ACTIVATE); pip install .
	$(MAKE) print_finish

dev: pre_install
	@. $(ACTIVATE); pip install -e .
	$(MAKE) print_finish

install_no_virt_env: $(REQ_FILE)
	@echo "\n\n****************************************"
	@echo "Installing NLP Architect in current python env"
	pip install -r $(REQ_FILE)
	pip install -e .
	@echo "NLP Architect setup complete."

sysinstall: $(REQ_FILE)
	@echo "\n\n****************************************"
	@echo "Installing NLP Architect in current python env (system install)"
	pip install -r $(REQ_FILE)
	pip install .
	@echo "NLP Architect setup complete."

print_finish:
	@echo "\n\n****************************************"
	@echo "Setup complete."
	@echo "Type:"
	@echo "    . '$(ACTIVATE)'"
	@echo "to work interactively $(VIRTUALENV_DIR) virtual env"
