#! /usr/bin/env bash

echo "Running flake8 check"
flake8 examples nlp_architect tests --config setup.cfg --output-file /tmp/nlp_architect_flake_out.txt
mkdir -p /tmp/nlp_architect_flake
echo "exporting to html"
flake8 examples nlp_architect tests --config setup.cfg --format=html --htmldir=/tmp/nlp_architect_flake
echo "html output can be found in /tmp/nlp_architect_flake/index.html"
echo "point your browser to that location to view status"
echo ""
echo "Running pylint check"
pylint -j 4 examples tests nlp_architect --rcfile=pylintrc --score=n
tee pylint.txt
echo ""
echo "Done."