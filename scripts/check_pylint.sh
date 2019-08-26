#! /usr/bin/env bash

function check_rc {
    RC="$?"
    if [[ RC -lt $1 ]] && [[ RC -gt 0 ]]
    then
        echo "RC = $RC < $1"
        echo "quitting ..."
        exit $?
    fi
}

echo "Running pylint check"
check_rc 1
pylint -j 4 examples tests nlp_architect --rcfile=pylintrc --score=n
check_rc 4
echo ""
echo "Done."

