#! /usr/bin/env bash

echo "Running NLP Architect tests"
pytest ./tests -rs -vv --cov=nlp_architect --junit-xml=pytest_unit.xml
echo "Done."