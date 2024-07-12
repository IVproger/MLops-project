import os
from datetime import datetime

from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.models.baseoperator import chain

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME")
PROJECT_ROOT = os.path.realpath(os.path.join(AIRFLOW_HOME, "../.."))
VENV_PATH = os.environ.get("VENV_PATH")


@dag(
    dag_id="data_prepare_dag",
    description="A DAG for data preparation. It runs ETL pipeline from ZenML after data_extract_dag successful run.",
    start_date=datetime(2024, 7, 1),
    schedule_interval="*/5 * * * *",  # Every 5 minutes
    catchup=False,
)
def data_prepare():
    # Wait for the data extraction workflow to complete
    sensor = ExternalTaskSensor(
        task_id="external_task_sensor",
        external_dag_id="data_extraction_v1_dag",
        external_task_id="load_to_datastore",
        timeout=600,  # Timeout after 10 minutes
        check_existence=True,  # Do not wait if there is no such task
    )

    # Execute the ZenML pipeline to prepare the data
    zenml_pipeline = BashOperator(
        task_id="run_zenml_data_prepare_pipeline",
        bash_command="python3 ./pipelines/data_prepare.py -prepare_data_pipeline ",
        cwd=PROJECT_ROOT,
        env={"PATH": os.path.join(VENV_PATH, "bin")},
        append_env=True,
    )

    # Run the ZenML pipeline after the data extraction workflow is successful
    chain(sensor, zenml_pipeline)


# Initialize the DAG
data_prepare_dag = data_prepare()
