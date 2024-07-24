import warnings

warnings.filterwarnings("ignore")

import os
import giskard.testing
import mlflow
import pandas as pd
from omegaconf import DictConfig
from src.model import scan_models_dir, balance_dataset
from src.data import extract_data, preprocess_data
from src.utils import init_hydra


def main():
    cfg = init_hydra("main")

    print("Iterating over models")
    for model_name, model_alias, model in scan_models_dir(cfg):
        print(f"Validating model {model_name} with alias {model_alias}")
        validate_model(cfg, model_name, model_alias, model)
    print("Finished validation")


def validate_model(
    cfg: DictConfig, model_name: str, model_alias: str, model: mlflow.pyfunc.PyFuncModel
):
    # Wrap dataset
    print("Wrapping dataset...")
    df, giskard_dataset, dataset_version = wrap_dataset(cfg)
    print(f"Dataset version: {dataset_version}")

    # Define the prediction function
    def predict_func(X: pd.DataFrame):
        return model.predict(X)

    # Create giskard model object
    print("Creating Giskard model object...")
    giskard_model = giskard.Model(
        model=predict_func,
        model_type="classification",
        data_preprocessing_function=lambda raw_df: preprocess_data(
            cfg, raw_df, require_target=False
        )[0],
        classification_labels=list(cfg.model.labels),
        feature_names=list(
            set(cfg.required) - {"Cancelled"}
        ),  # Validate only the required features
        name=model_name,
        classification_threshold=0.5,
    )

    # Run model scanning and generate html report
    print("Running model scan...")
    scan_model(
        giskard_model,
        giskard_dataset,
        f"validation_results_{model_name}_{model_alias}_{cfg.data.dataset_name}_{dataset_version}.html",
    )

    # Run test suite
    print("Running model tests...")
    test_results = run_model_tests(
        cfg.model.f1_threshold,
        giskard_model,
        giskard_dataset,
        f"test_suite_{model_name}_{model_alias}_{cfg.data.dataset_name}_{dataset_version}",
    )
    if test_results.passed:
        print("Passed model validation!")
    else:
        print("Model has vulnerabilities!")


def wrap_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, giskard.Dataset, str]:
    """
    Wrap raw dataset into Giskard Dataset object.
    """
    # Load data into data frame
    df, version = extract_data(version=cfg.test_data_version, cfg=cfg)
    df = balance_dataset(df)

    # Wrap data into Giskard Dataset object
    giskard_dataset = giskard.Dataset(
        df=df,
        target=cfg.data.target_cols[0],
        name=cfg.data.dataset_name,
    )

    return df, giskard_dataset, version


def scan_model(model: giskard.Model, dataset: giskard.Dataset, report_name: str):
    """
    Scan model using Giskard and save the results in html file.
    """
    # Run the scan
    scan_results = giskard.scan(model, dataset)

    # Save the results in html file
    scan_results.to_html(os.path.join("reports", report_name))


def run_model_tests(
    f1_threshold: float, model: giskard.Model, dataset: giskard.Dataset, suite_name: str
):
    """
    Create test suite using Giskard and run the tests.
    """
    # Declare test suite
    test_suite = giskard.Suite(name=suite_name)
    test1 = giskard.testing.test_f1(
        # TODO: F1 weighted
        model=model,
        dataset=dataset,
        threshold=f1_threshold,
    )

    # Run the test suite
    test_suite.add_test(test1)
    test_results = test_suite.run()
    return test_results


if __name__ == "__main__":
    main()
