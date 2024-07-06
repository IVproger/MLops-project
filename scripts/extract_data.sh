#!/bin/bash

# Fail the script if any command fails
set -e

# Take a new sample and validate it
python3 -m src.data
