#! /usr/bin/env bash

function check_rc {
    RC="$?"
    if [[ RC -gt $1 ]]
    then
        echo "RC = $RC > $1"
        echo "quitting ..."
        exit $?
    fi
}

echo "Running flake8 check"
flake8 examples nlp_architect tests solutions --config setup.cfg --output-file /tmp/nlp_architect_flake_out.txt
check_rc 0
mkdir -p /tmp/nlp_architect_flake
echo "exporting to html"
flake8 examples nlp_architect tests solutions --config setup.cfg --format=html --htmldir=/tmp/nlp_architect_flake
check_rc 0
echo "html output can be found in /tmp/nlp_architect_flake/index.html"
echo "point your browser to that location to view status"
echo "Done."

