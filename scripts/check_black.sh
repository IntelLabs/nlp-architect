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

echo "Running black check"
black --check --line-length 100 --target-version py36 examples nlp_architect solutions tests
check_rc 0
echo "Done running black"