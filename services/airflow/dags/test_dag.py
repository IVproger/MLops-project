import os

from pendulum import datetime
from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from airflow.models.baseoperator import chain

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME")
PROJECT_ROOT = os.path.join(AIRFLOW_HOME, "../..")
VENV_PATH = os.environ.get("VENV_PATH")
PATH = os.environ.get("PATH")


@dag(
    dag_id="test",
    description="A DAG for data extraction, validation, versioning, and loading",
    start_date=datetime(2024, 7, 1, tz="UTC"),
    schedule_interval="*/15 * * * *",
    catchup=False,
)
def data_extraction_workflow():
    push_to_git = BashOperator(
        task_id="pust_to_git",
        bash_command="scripts/push_sample_version.sh ",
        cwd=PROJECT_ROOT,
        env={"PATH": os.path.join(VENV_PATH, "bin") + ":" + PATH},
        append_env=True,
    )

    chain(push_to_git)


data_extraction_workflow = data_extraction_workflow()
