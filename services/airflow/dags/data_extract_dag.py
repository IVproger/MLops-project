from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
import os

with DAG(
    "data_extraction_workflow",
    start_date=datetime(2024, 1, 1),
    description="A DAG for data extraction, validation, versioning, and loading",
    schedule_interval="*/5 * * * *",
    catchup=False,
) as dag:
    airflow_home = os.environ.get("AIRFLOW_HOME")

    extract_data = BashOperator(
        task_id="extract_data",
        bash_command=f"cd {airflow_home} && cd project && python3 -m src.data",
    )

    load_to_datastore = BashOperator(
        task_id="load_to_datastore",
        bash_command=f"cd {airflow_home} && cd project && dvc add data/samples/sample.csv && dvc push",
    )

    extract_data >> load_to_datastore
