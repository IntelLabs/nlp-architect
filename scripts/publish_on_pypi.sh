#!/bin/bash

# Publishing NLP Architect on pypi script
# =======================================
#
# Steps:
# 1 - make sure you install setuptools, wheel and twine python packages:
#
#   pip3 install -U setuptools wheel twine
#
# 2 - Update NLP Architect version in nlp_architect.version
# 3 - Run nlp_architect test/style and make sure everything passes
# 4 - Clear build/ and dist/ and nlp_architect.egg-info/ directories
#
#   rm -rf build/ dist/ nlp_architect.egg-info/
#
# 5 - Compile NLP Architect source and wheel (output in dist/:
#
#   python3 setup.py sdist bdist_wheel
#
# optional - Upload to test.pypi
#
#   twine upload --repository-url https://test.pypi.org/legacy/ dist/*
#
# 6 - Upload to pypi:
#
#   twine upload dist/*
#

pip3 install -U setuptools wheel twine
rm -rf build/ dist/ nlp_architect.egg-info/
python3 setup.py sdist bdist_wheel
twine upload dist/*