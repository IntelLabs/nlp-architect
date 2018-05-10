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

FLAKE8_CHECK_DIRS :=
PYLINT_CHECK_DIRS := *
DOC_DIR := doc
DOC_PUB_RELEASE_PATH := $(DOC_PUB_PATH)/$(RELEASE)

LIBRARY_NAME := nlp_architect
VIRTUALENV_DIR := .nlp_architect_env
GEN_REQ_FILE := _generated_reqs.txt
ACTIVATE := $(VIRTUALENV_DIR)/bin/activate
MODELS_DIR := $(LIBRARY_NAME)/models
NLP_DIR := $(LIBRARY_NAME)/nlp

.PHONY: test_prepare style fixstyle autopep8 doc_prepare doc html clean \
	install install_dev install_no_virt_env $(ACTIVATE)

default: install_dev

test_prepare: test_requirements.txt
	pip install -r test_requirements.txt > /dev/null 2>&1

style: test_prepare
	flake8 --exit-zero --output-file flake.txt --tee $(FLAKE8_CHECK_DIRS)
	pylint --reports=n --output-format=colorized --ignore=.venv $(PYLINT_CHECK_DIRS) || true

fixstyle: autopep8

autopep8:
	autopep8 -a -a --global-config setup.cfg --in-place `find . -name \*.py`
	echo run "git diff" to see what may need to be checked in and "make style" to see what work remains

doc_prepare: doc_requirements.txt
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

clean:
	@echo "Cleaning files.."
	@rm -rf $(VIRTUALENV_DIR)
	@rm -rf $(GEN_REQ_FILE)

$(ACTIVATE):
	@echo "NLP Architect installation"
	@echo "**************************"
	@echo "creating new environment"
	virtualenv -p python3 $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip install -U pip

pre_install: $(ACTIVATE) generate_reqs.sh
	@echo "\n\n****************************************"
	@echo "Generating package list to install"
	@. $(ACTIVATE); bash generate_reqs.sh
	@echo "Installing packages ..."
	@. $(ACTIVATE); pip install -r $(GEN_REQ_FILE)

install: pre_install
	@. $(ACTIVATE); pip install .
	$(MAKE) print_finish

install_dev: pre_install
	@. $(ACTIVATE); pip install -e .
	$(MAKE) print_finish

install_no_virt_env:
	@echo "\n\n****************************************"
	@echo "Installing NLP Architect in current python env"
	@echo "Generating package list to install"
	bash generate_reqs.sh
	pip install -r $(GEN_REQ_FILE)
	@echo "Installation done"

print_finish:
	@echo "\n\n****************************************"
	@echo "Setup complete."
	@echo "Type:"
	@echo "    . '$(ACTIVATE)'"
	@echo "to work interactively $(VIRTUALENV_DIR) virtual env"
