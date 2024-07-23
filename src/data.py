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


def sample_data(cfg: DictConfig):
    """
    The function to sample the data from the given URL and save it to the sample path.
    Returns both the sampled data and the updated configuration settings without updating the real config files.
    """
    try:
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
            else cfg.data.last_included_row_number + 1
        )

        # If the start_row + sample_size exceeds total_rows, start from the beginning
        if start_row + sample_size > total_rows:
            start_row = 0  # Reset to start from the beginning

        # Load only the necessary rows into memory
        skiprows = range(
            1, start_row + 1
        )  # Skip rows before the start_row, keeping header
        nrows = sample_size  # Number of rows to read
        data = pd.read_csv(datastore_path, skiprows=skiprows, nrows=nrows)

        print("Sampling data...")
        resulted_sample = data

        # Take the representative sample of 50/50 % of classes instances
        cancelled = resulted_sample[resulted_sample["Cancelled"]]
        on_time = resulted_sample[~resulted_sample["Cancelled"]]

        representative_persent = (cancelled.shape[0] * 100 / on_time.shape[0]) / 100

        on_time = on_time.sample(frac=representative_persent)

        df = pd.concat([cancelled, on_time])

        # Create a deep copy of cfg to modify without affecting the original
        updated_cfg = copy.deepcopy(cfg)

        # Update the configuration for last included row number in the copy
        new_last_included_row_number = start_row + sample_size - 1
        updated_cfg.data.last_included_row_number = (
            new_last_included_row_number % total_rows
        )

        # Increment and update the data version in the copy
        new_version = f"v{updated_cfg.data.version_number+1}.0"
        updated_cfg.data.data_version = new_version
        updated_cfg.data.version_number = updated_cfg.data.version_number + 1

        # Return both the sampled data and the updated configuration
        return df, updated_cfg
    except Exception as e:
        print("Error in loading or sampling the data: ", e)
        return None, cfg


def validate_initial_data(cfg: DictConfig, df: pd.DataFrame):
    """
    Validate the data using Great Expectations.
    """
    try:
        context = gx.get_context(
            project_root_dir=cfg.data.context_dir_path, mode="file"
        )
        ds: PandasDatasource = context.sources.add_or_update_pandas(name="sample_data")
        ds.add_dataframe_asset(
            name="sample_file",
            dataframe=df,
        )
        checkpoint = context.get_checkpoint("sample_checkpoint")

        checkpoint_result = checkpoint.run()

        if checkpoint_result.success:
            print("Validation successful.")
            return True
        else:
            print("Validation failed.")
            print(checkpoint_result)
    except Exception as e:
        print("Error in validating the data: ", e)
    return False


def read_datastore() -> tuple[pd.DataFrame, str]:
    """
    Read the sample data.
    """
    cfg = OmegaConf.load("configs/data_sample.yaml")
    data = pd.read_csv(cfg.data.sample_path)
    version = open("configs/data_version.txt", "r").read().strip()
    return data, version


def preprocess_data(
    cfg: DictConfig, df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the data and return features and target
    """

    # Drop NaNs
    df = df.dropna(axis=1)

    # Split the dataset into X and y
    X = df.drop(["Cancelled"], axis=1)
    y = df[["Cancelled"]]

    # Feature extractor
    feature_extractor = (
        "feature_extractor",
        dtf.feature_extractor(cfg.required),
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

    # Transformers for hashing
    hash_transformers = [
        (f"hash_{feature}", dtf.hash_transformer(feature))
        for feature in df.columns[df.dtypes == "object"]
    ]

    # Assembling the pipeline
    pipeline = Pipeline(
        [
            feature_extractor,
            *cyclic_transformers,
            *hhmm_transformers,
            *hash_transformers,
        ]
    )

    return pipeline.transform(X), y


def validate_features(
    X: pd.DataFrame, y: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate the data using Great Expectations.
    """
    cfg = OmegaConf.load("configs/data_sample.yaml")
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
        print("Validation failed.")
        print(checkpoint_result)

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
