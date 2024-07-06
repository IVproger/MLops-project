"""
Import the necessary libraries and modules for the data sampling script.
"""

import os
import gdown
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import copy
import great_expectations as gx
from great_expectations.datasource.fluent import PandasDatasource


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
            gdown.download(cfg.data.url, cfg.data.output, quiet=False)

        # Read the data from the source
        print("Download data into frame...")
        data = pd.read_csv(datastore_path)

        # Initialize start_row to 0 if last_included_row_number is not set or indicates no rows have been included yet
        start_row = (
            0
            if cfg.data.last_included_row_number < 0
            else cfg.data.last_included_row_number + 1
        )
        total_rows = len(data)
        sample_size = int(total_rows * cfg.data.sample_size)

        print("Sampling data...")
        resulted_sample = data.iloc[start_row : start_row + sample_size]

        # Create a deep copy of cfg to modify without affecting the original
        updated_cfg = copy.deepcopy(cfg)

        # Update the configuration for last included row number in the copy
        updated_cfg.data.last_included_row_number = start_row + sample_size - 1

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
        print(df.head())
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


def read_datastore():
    pass


def preprocess_data():
    pass


def validate_features():
    pass


def load_features():
    pass


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
