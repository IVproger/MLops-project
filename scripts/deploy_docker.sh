#!/bin/bash

docker compose build ml
docker compose up -d ml
docker compose build ml --push
