from pendulum import datetime
from airflow.decorators import dag, task
import os
from omegaconf import OmegaConf
from airflow.exceptions import AirflowFailException

from src.data import sample_data

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME")
PROJECT_ROOT = os.path.join(AIRFLOW_HOME, "project")
config_path = os.path.join(PROJECT_ROOT, "configs/main.yaml")
data_version_path = os.path.join(PROJECT_ROOT, "configs/data_version.txt")


@dag(
    dag_id="data_extraction_workflow",
    description="A DAG for data extraction, validation, versioning, and loading",
    start_date=datetime(2024, 7, 1, tz="UTC"),
    schedule_interval="*/5 * * * *",
    catchup=False,
)
def data_extraction_workflow():
    @task
    def extract_data():
        cfg = OmegaConf.load(config_path)
        print(cfg)
        extracted_data, new_cfg = sample_data(cfg)
        if extracted_data is not None:
            # Convert DictConfig to dict for serialization
            new_cfg_dict = OmegaConf.to_container(new_cfg, resolve=True)
            return {"extracted_data": extracted_data, "cfg": new_cfg_dict}
        else:
            raise AirflowFailException("Extraction returned None, failing the task.")

    extract_data()

    # @task
    # def validate_data(data):
    #     extracted_data = data["extracted_data"]
    #     cfg = OmegaConf.create(data["cfg"])
    #     if validate_initial_data(cfg):
    #         OmegaConf.save(cfg, config_path)
    #         extracted_data.to_csv(cfg.data.sample_path, index=False)
    #         with open(data_version_path, "w", encoding="utf-8") as file:
    #             file.write(str(cfg.data.data_version) + "\n")
    #         return {"extracted_data": extracted_data, "cfg": OmegaConf.to_container(cfg, resolve=True)}
    #     else:
    #         raise AirflowFailException("Validation failed, failing the task.")

    # data = extract_data()
    # validate_data(data)

    # version_the_sample = BashOperator(
    #     task_id="version_the_data_sample",
    #     bash_command="dvc add {{ params.sample_path }} && dvc commit -m 'Versioning sample data'",
    #     cwd=PROJECT_ROOT,
    #     params={"sample_path": "data/samples/sample.csv"},
    # )

    # load_to_datastore = BashOperator(
    #     task_id="load_to_datastore",
    #     bash_command="dvc push",
    #     cwd=PROJECT_ROOT,
    # )

    # chain(data, version_the_sample, load_to_datastore)


data_extraction_workflow = data_extraction_workflow()
