#!/bin/bash

echo "Cleaning old docs"
make -C ../docs-source clean
echo "Creating new docs"
make -C ../docs-source html
echo "Serving website on localhost:8000"
cd ../docs-source/build/html; python3 -m http.server
