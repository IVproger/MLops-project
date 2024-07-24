#!/bin/bash

# Fail the script if any command fails
set -e

docker compose build ml
docker compose up -d ml
docker compose build ml --push
