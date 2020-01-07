#! /usr/bin/env bash

echo "Running NLP Architect tests"
# pytest ./ -rs -vv --cov=nlp_architect --junit-xml=pytest_unit.xml
pytest ./ -rs -n 8 --dist=loadfile
echo "Done."