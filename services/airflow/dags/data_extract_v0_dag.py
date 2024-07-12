import os
from datetime import datetime

from airflow.decorators import dag
from airflow.operators.bash import BashOperator

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME")
PROJECT_ROOT = os.path.join(AIRFLOW_HOME, "../..")
VENV_PATH = os.environ.get("VENV_PATH")


@dag(
    "data_extraction_v0_dag",
    description="A DAG for extraction of a new sample, validation, versioning using DVC, and loading to DVC data store.",
    start_date=datetime(2024, 7, 1),
    schedule_interval="*/5 * * * *",  # Every 5 minutes
    catchup=False,
)
def data_extract():
    # Extract data, validate it, and version it
    extract_data = BashOperator(
        task_id="extract_data",
        bash_command="echo $PATH && python3 -m src.data",
        cwd=PROJECT_ROOT,
        env={"PATH": os.path.join(VENV_PATH, "bin")},
        append_env=True,
    )

    # Load the data to the DVC data store
    load_to_datastore = BashOperator(
        task_id="load_to_datastore",
        bash_command="dvc add data/samples/sample.csv && dvc push",
        cwd=PROJECT_ROOT,
        env={"PATH": os.path.join(VENV_PATH, "bin")},
        append_env=True,
    )

    # Define the task dependencies
    extract_data >> load_to_datastore


# Initialize the DAG
data_extraction_dag = data_extract()
