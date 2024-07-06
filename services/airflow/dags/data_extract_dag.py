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

    run_test_sh = BashOperator(
        task_id="extract_data_sh",
        bash_command=f"cd {airflow_home} && cd ../../ && python3 -m src.data",
    )

    run_push_sample_version_sh = BashOperator(
        task_id="push_sample_version.sh",
        bash_command=f"cd {airflow_home} && cd ../../ && scripts/push_sample_version.sh ",
    )

    run_test_sh >> run_push_sample_version_sh
