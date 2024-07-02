#!/bin/bash

# Fail the script if any command fails
set -x

python3 -m src.data
dvc add data/samples/sample.csv
dvc push
