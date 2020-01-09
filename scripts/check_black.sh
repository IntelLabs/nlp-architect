#! /usr/bin/env bash
echo "Running black check"
black --check --line-length 100 --target-version py36 examples nlp_architect solutions tests
echo "Done running black"