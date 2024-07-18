import warnings

warnings.filterwarnings("ignore")

import os
import giskard.testing
import mlflow
import pandas as pd
from omegaconf import DictConfig
from src.model import retrieve_model_with_alias
from src.data import extract_data, preprocess_data
from src.utils import init_hydra


def main():
    cfg = init_hydra("main")

    # Wrap dataset and model
    print("Wrapping dataset and model...")
    df, giskard_dataset, dataset_version = wrap_dataset(cfg)
    model, model_version = wrap_model(cfg)
    print(f"Dataset version: {dataset_version}")
    print(f"Model version: {model_version}")

    # Create giskard model object
    print("Creating Giskard model object...")
    predict_func = get_predict_func(cfg, model)
    giskard_model = giskard.Model(
        model=predict_func,
        model_type="classification",
        data_preprocessing_function=lambda raw_df: preprocess_data(
            cfg, raw_df, require_target=False
        )[0],
        classification_labels=list(cfg.model.labels),
        # The order MUST be identical to the prediction_function's output order
        feature_names=cfg.required,  # By default, all columns of the passed dataframe
        name=cfg.model.best_model_name,  # Optional: give it a name to identify it in metadata
        # classification_threshold=0.5, # Optional: Default: 0.5
    )

    # Run model scanning and generate html report
    print("Running model scan...")
    scan_model(
        cfg,
        giskard_model,
        giskard_dataset,
        f"validation_results_{cfg.model.best_model_name}_{model_version}_{cfg.data.dataset_name}_{dataset_version}.html",
    )

    # Run test suite
    print("Running model tests...")
    test_results = run_model_tests(
        cfg,
        giskard_model,
        giskard_dataset,
        f"test_suite_{cfg.model.best_model_name}_{model_version}_{cfg.data.dataset_name}_{dataset_version}",
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

    # Wrap data into Giskard Dataset object
    giskard_dataset = giskard.Dataset(
        df=df,
        target=cfg.data.target_cols[0],
        name=cfg.data.dataset_name,
    )

    return df, giskard_dataset, version


def wrap_model(cfg: DictConfig) -> tuple[mlflow.pyfunc.PyFuncModel, int]:
    """
    Fetch model from the MLFlow model registry.
    """
    model_name = cfg.model.best_model_name
    model_alias = cfg.model.best_model_alias

    # Load model
    model: mlflow.pyfunc.PyFuncModel = retrieve_model_with_alias(
        model_name, model_alias=model_alias
    )

    # Get its version
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(name=model_name, alias=model_alias)
    model_version = mv.version

    return model, model_version


def get_predict_func(cfg: DictConfig, model: mlflow.pyfunc.PyFuncModel):
    """
    Custom predict function that preprocesses data and then predicts using the model.
    """

    def predict_func(X: pd.DataFrame):
        # X, _ = preprocess_data(cfg, raw_df, validate_target=False)
        return model.predict(X)

    return predict_func


def scan_model(
    cfg: DictConfig, model: giskard.Model, dataset: giskard.Dataset, report_name: str
):
    """
    Scan model using Giskard and save the results in html file.
    """
    # Run the scan
    scan_results = giskard.scan(model, dataset)

    # Save the results in html file
    scan_results.to_html(os.path.join("reports", report_name))


def run_model_tests(
    cfg: DictConfig, model: giskard.Model, dataset: giskard.Dataset, suite_name: str
):
    """
    Create test suite using Giskard and run the tests.
    """
    # Declare test suite
    test_suite = giskard.Suite(name=suite_name)
    test1 = giskard.testing.test_f1(
        model=model, dataset=dataset, threshold=cfg.model.f1_threshold
    )

    # Run the test suite
    test_suite.add_test(test1)
    test_results = test_suite.run()
    return test_results


if __name__ == "__main__":
    main()
