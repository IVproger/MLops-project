#!/bin/bash

mlflow run . --env-manager=local -e predict -P example_version=v1.0
mlflow run . --env-manager=local -e predict -P example_version=v2.0
mlflow run . --env-manager=local -e predict -P example_version=v3.0
mlflow run . --env-manager=local -e predict -P example_version=v4.0
mlflow run . --env-manager=local -e predict -P example_version=v5.0
