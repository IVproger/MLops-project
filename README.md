# MLOps Capstone project

[![Code linting](https://github.com/IVproger/MLops-project/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/IVproger/MLops-project/actions/workflows/pre-commit.yaml)
[![Code testing](https://github.com/IVproger/MLops-project/actions/workflows/test-code.yaml/badge.svg)](https://github.com/IVproger/MLops-project/actions/workflows/test-code.yaml)
[![Model validation](https://github.com/IVproger/MLops-project/actions/workflows/validate-model.yaml/badge.svg)](https://github.com/IVproger/MLops-project/actions/workflows/validate-model.yaml)

Presentation: https://drive.google.com/drive/folders/1NntH7FiwwmKuPwTDOSu5uE7WJ73fX7xn

DockerHub ML image: https://hub.docker.com/r/artemsbulgakov/mlops-model

## Predicting on-time, delayed and cancelled flights

> Business problem revolves around improving operational efficiency and passenger satisfaction through accurate flight
> status predictions. By classifying flights as on-time, canceled, or diverted, airlines can gain crucial insights to
> optimize their operations and provide a better service to their passengers.

Dataset: https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022 (Combined_Flights_2022.csv)

Team members:

- Ivan Golov
- Artem Bulgakov
- Alexey Tkachenko

## Deploy best model

1. Install Docker (or Docker Desktop) with Docker Compose plugin.
2. Run Flask API and Gradio UI:
   ```bash
   docker compose up --build ml-gradio ml-api ml
   ```
3. Check Gradio UI at http://localhost:8084, Flask API at http://localhost:8083, model API at http://localhost:8082.

## Setup

We use Docker to setup Airflow and run ZenML server. For local development and running MLFlow, we use Poetry venv.

### Setup Docker

1. Install Docker (or Docker Desktop) with Docker Compose plugin.
2. Run the following command:
    ```bash
    mkdir -p ./services/airflow/dags ./services/airflow/logs ./services/airflow/plugins ./services/airflow/config
    echo -e "AIRFLOW_UID=$(id -u)" > .env
    ```
3. Build images:
   ```bash
   docker compose build --pull
   ```

### Setup Poetry

1. Install Python 3.11
2. Install Poetry 1.8.3
3. Add poetry plugin
   ```bash
   $ poetry self add poetry-plugin-export
   ```
4. Configure `.venv` location
   ```bash
   $ poetry config virtualenvs.in-project true
   ```
5. Create `.venv` with Python 3.11 (make sure you have it installed)
   ```bash
   $ poetry env use python3.11
   ```
6. Install dependencies
   ```bash
   $ poetry install --with dev
   ```
7. Set up pre-commit hooks
   ```bash
   $ poetry run pre-commit install --install-hooks -t pre-commit -t commit-msg
   ```

## Running services

### Airflow and ZenML

We use Docker Compose to run all services of Airflow and ZenML server.

1. Start Airflow services and ZenML server:
   ```bash
   docker compose up --build
   ```
2. Wait several minutes (Airflow may take a lot of time to start).
3. Access Airflow at http://localhost:8080 (default login `airflow`, password `airflow`).
4. Access ZenML server at http://localhost:8081 (default login `admin`, password `admin`).

### Running MLFlow

1. Start MLFlow server in one terminal:
   ```bash
   mlflow server
   ```
2. Run entry point in second terminal:
   ```bash
   mlflow run . --env-manager=local
   ```
3. Wait for all models to train.
4. Access MLFlow server at http://localhost:5000.

## Docs
Each folder contains a `README.md` file with short description for every file. Code is well-documented with inline comments and representative symbol    names
