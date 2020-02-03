#! /usr/bin/env bash

echo "Running NLP Architect tests"
python -m spacy download en
pytest ./ -rs -v -n 20 --dist=loadfile --cov=nlp_architect --junit-xml=pytest_unit.xml
echo "Done."