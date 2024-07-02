#!/bin/bash

# Set the path to the .venv/bin/activate file
activate_file=".venv/bin/activate"

sed -i '/deactivate () {/a\
    unset AIRFLOW_HOME' "$activate_file"

# Env vars
echo "export AIRFLOW_HOME=\$PWD/services/airflow" >> $activate_file
echo "export PYTHONPATH=\$PWD/src" >> $activate_file

# Restart venv
echo ".venv/bin/activate is extended. Restart your terminal"