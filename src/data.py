"""
Import the necessary libraries and modules for the data sampling script.
"""

import os
import copy
from pathlib import Path
import math
import gdown
import pandas as pd
from sklearn.pipeline import Pipeline
import zenml
from omegaconf import DictConfig, OmegaConf
import great_expectations as gx
from great_expectations.datasource.fluent import PandasDatasource
from src import data_transformations as dtf
import dvc.api

from src.utils import init_hydra


def sample_data(cfg: DictConfig):
    """
    The function to sample the data from the given URL and save it to the sample path.
    Returns both the sampled data and the updated configuration settings without updating the real config files.
    """
    datastore_path = cfg.data.datastore_path

    # Create datastore directory if not exists
    Path(datastore_path).parent.mkdir(exist_ok=True, parents=True)

    # Check if the source data is available, if not download it.
    if not os.path.exists(datastore_path):
        print("Downloading data from: ", cfg.data.url)
        gdown.download(cfg.data.url, datastore_path, quiet=False, use_cookies=False)

    # Determine the total number of rows in the file without loading it entirely
    total_rows = sum(1 for row in open(datastore_path, "r")) - 1  # Exclude header

    # Calculate the sample size
    sample_size = math.ceil(total_rows * cfg.data.sample_size)

    # Determine the start row for sampling
    start_row = (
        0
        if cfg.data.last_included_row_number < 0
        else (cfg.data.last_included_row_number + 1) % total_rows
    )

    # If the start_row + sample_size exceeds total_rows, adjust the sample size
    if start_row + sample_size > total_rows:
        sample_size = total_rows - start_row

    # Load only the necessary rows into memory
    skiprows = range(1, start_row + 1)  # Skip rows before the start_row, keeping header
    nrows = sample_size  # Number of rows to read
    data = pd.read_csv(datastore_path, skiprows=skiprows, nrows=nrows)

    print("Sampling data...")
    resulted_sample = data

    # Create a deep copy of cfg to modify without affecting the original
    updated_cfg = copy.deepcopy(cfg)

    # Update the configuration for last included row number in the copy
    new_last_included_row_number = start_row + sample_size - 1
    updated_cfg.data.last_included_row_number = (
        new_last_included_row_number % total_rows
    )

    # Increment and update the data version in the copy
    new_version = f"v{updated_cfg.data.version_number + 1}.0"
    updated_cfg.data.data_version = new_version
    updated_cfg.data.version_number = updated_cfg.data.version_number + 1

    # Return both the sampled data and the updated configuration
    return resulted_sample, updated_cfg


def validate_initial_data(cfg: DictConfig, df: pd.DataFrame):
    """
    Validate initial data using Great Expectations.
    """
    context = gx.get_context(project_root_dir=cfg.data.context_dir_path, mode="file")
    ds: PandasDatasource = context.sources.add_or_update_pandas(name="sample_data")
    ds.add_dataframe_asset(
        name="sample_file",
        dataframe=df,
    )
    checkpoint = context.get_checkpoint("sample_checkpoint")

    checkpoint_result = checkpoint.run()

    if checkpoint_result.success:
        print("Validation successful.")
    else:
        raise ValueError("Validation failed: ", checkpoint_result)


def read_datastore() -> tuple[pd.DataFrame, str]:
    """
    Read the sample data.
    """
    cfg = OmegaConf.load("configs/data_sample.yaml")
    data = pd.read_csv(cfg.data.sample_path)
    version = open("configs/data_version.txt", "r").read().strip()
    return data, version


def preprocess_data(
    cfg: DictConfig, df: pd.DataFrame, require_target: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the data and return features and target
    """

    # Drop NaNs
    df = df.dropna(axis=1)

    # Split the dataset into X and y
    required: list[str] = copy.deepcopy(cfg.required)
    if require_target or "Cancelled" in df.columns:
        X = df.drop(["Cancelled"], axis=1)
        y = df[["Cancelled"]]
    else:
        X = df
        y = None
        if "Cancelled" in required:
            required.remove("Cancelled")

    # Feature extractor
    feature_extractor = (
        "feature_extractor",
        dtf.feature_extractor(required),
    )

    # Transformers for cyclic data
    cyclic_transformers = [
        (f"cyclic_{feature}", dtf.cyclic_transformer(feature, period))
        for (feature, period) in cfg.time_features
    ]

    # Transformers for HHMM data
    hhmm_transformers = [
        (f"hhmm_{feature}", dtf.hhmm_transformer(feature)) for feature in cfg.hhmm
    ]

    # Assembling the pipeline
    pipeline = Pipeline(
        [
            feature_extractor,
            *cyclic_transformers,
            *hhmm_transformers,
        ]
    )

    X = pipeline.transform(X)

    # Transformers for hashing
    hash_transformers = [
        (f"hash_{feature}", dtf.hash_transformer(feature))
        for feature in X.columns[X.dtypes == "object"]
    ]

    pipeline = Pipeline(
        [
            *hash_transformers,
        ]
    )

    return pd.DataFrame(pipeline.transform(X)), y


def validate_features(
    X: pd.DataFrame, y: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate the data post-transformation using Great Expectations.
    """
    cfg = init_hydra("main")
    context = gx.get_context(project_root_dir=cfg.data.context_dir_path, mode="file")
    ds: PandasDatasource = context.sources.add_or_update_pandas(
        name="transformed_sample"
    )
    ds.add_dataframe_asset(name="X", dataframe=X)
    ds.add_dataframe_asset(name="y", dataframe=y)

    checkpoint = context.get_checkpoint("transformations_checkpoint")

    checkpoint_result = checkpoint.run()

    if checkpoint_result.success:
        print("Validation successful.")
    else:
        raise Exception("Validation failed: ", checkpoint_result)

    return X, y


def load_features(X: pd.DataFrame, y: pd.DataFrame, version: str) -> None:
    """
    Save the features and target as artifact.
    """
    zenml.save_artifact(data=[X, y], name="features_target", tags=[version])


def extract_data(version: str, cfg: DictConfig):
    with dvc.api.open(cfg.data.sample_path, rev=version) as fd:
        return pd.read_csv(fd), version


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/data_sample.yaml")

    # Take a new sample
    sample, new_cfg = sample_data(cfg)

    if sample is None:
        raise ValueError("Data sampling failed. Exiting...")

    # Validate the data
    if not validate_initial_data(cfg, sample):
        raise ValueError("Data validation failed. Exiting...")

    # Save the generated sample of data and new configuration settings
    sample.to_csv(cfg.data.sample_path, index=False)
    OmegaConf.save(new_cfg, "configs/data_sample.yaml")
    with open("configs/data_version.txt", "w", encoding="utf-8") as file:
        file.write(new_cfg.data.data_version + "\n")
