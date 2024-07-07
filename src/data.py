"""
Import the necessary libraries and modules for the data sampling script.
"""

import os
import copy
import math

import gdown
import pandas as pd
import zenml
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
import great_expectations as gx
from great_expectations.datasource.fluent import PandasDatasource

from src import data_transformations as dtf


def sample_data(cfg: DictConfig):
    """
    The function to sample the data from the given URL and save it to the sample path.
    Returns both the sampled data and the updated configuration settings without updating the real config files.
    """
    try:
        datastore_path = cfg.data.datastore_path

        # Check if the source data is available, if not download it.
        if not os.path.exists(datastore_path):
            print("Downloading data from: ", cfg.data.url)
            gdown.download(cfg.data.url, datastore_path, quiet=False)

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
        return resulted_sample, updated_cfg
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
    cfg = OmegaConf.load("configs/main.yaml")
    data = pd.read_csv(cfg.data.sample_path)
    version = open("configs/data_version.txt", "r").read().strip()
    return data, version


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the data to extract features and target.
    """
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="data_transformations")

        # 1. Drop unnecessary features
        df = dtf.pull_features(df, cfg["required"])

        # 2. Hash string features
        for c in cfg["hash_features"]:
            df = dtf.hash_feature(df, c)

        # 3. Fix time values and encode them as cyclic features
        for c in cfg["hhmm"]:
            df, colHH, colMM = dtf.fix_hhmm(df, c)
            df = dtf.encode_cyclic_time_data(df, colHH, 24)
            df = dtf.encode_cyclic_time_data(df, colMM, 60)

        # 4. Encode remaining cyclic features
        for tf in cfg["time_features"]:
            print(tf)
            df = dtf.encode_cyclic_time_data(df, tf[0], tf[1])
        
        # 5. Split the dataset into X and y
        X = df.drop(["Cancelled"], axis=1)
        y = df["Cancelled"]
        return X, y


def validate_features(
    X: pd.DataFrame, y: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate the features and target using Great Expectations.
    """
    # TODO: Implement feature validation
    return X, y


def load_features(X: pd.DataFrame, y: pd.DataFrame, version: str) -> None:
    """
    Save the features and target as artifact.
    """
    zenml.save_artifact(data=[X, y], name="features_target", tags=[version])


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/main.yaml")

    # Take a new sample
    sample, new_cfg = sample_data(cfg)

    if sample is None:
        raise ValueError("Data sampling failed. Exiting...")

    # Validate the data
    if not validate_initial_data(cfg, sample):
        raise ValueError("Data validation failed. Exiting...")

    # Save the generated sample of data and new configuration settings
    sample.to_csv(cfg.data.sample_path, index=False)
    OmegaConf.save(new_cfg, "configs/main.yaml")
    with open("configs/data_version.txt", "w", encoding="utf-8") as file:
        file.write(new_cfg.data.data_version + "\n")
