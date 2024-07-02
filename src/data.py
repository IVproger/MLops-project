"""
Import the necessary libraries and modules for the data sampling script.
"""

import os
import gdown
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from src.data_quality import load_context_and_sample_data, define_expectations


@hydra.main(version_base=None, config_path="configs", config_name="main")
def sample_data(cfg: DictConfig):
    """
    The function to sample the data from the given URL and save it to the sample path.
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

        # Update the configuration for last included row number
        cfg.data.last_included_row_number = start_row + sample_size - 1

        # Increment and update the data version
        new_version = f"v{cfg.data.version_number+1}.0"
        cfg.data.data_version = new_version
        cfg.data.version_number = cfg.data.version_number + 1

        # Save the updated configuration back to the YAML file
        OmegaConf.save(config=cfg, f="configs/main.yaml")

        with open("configs/data_version.txt", "w", encoding="utf-8") as f:
            f.write(new_version)

        return resulted_sample
    except Exception as e:
        print("Error in loading or sampling the data: ", e)
        return None


@hydra.main(version_base=None, config_path="configs", config_name="main")
def validate_initial_data(cfg: DictConfig):
    """
    Validate the data using Great Expectations.
    """
    try:
        context, da = load_context_and_sample_data("../services", cfg.data.sample_path)
        batch_request = da.build_batch_request()
        validator = define_expectations(context, batch_request)
        validator.save_expectation_suite(discard_failed_expectations=False)
        checkpoint = context.add_or_update_checkpoint(
            name="sample_checkpoint",
            validator=validator,
        )
        checkpoint_result = checkpoint.run()

        if checkpoint_result.success:
            print("Validation failed.")
            print(checkpoint_result)
        else:
            print("Validation successful.")
    except Exception as e:
        print("Error in validating the data: ", e)


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
    sample = sample_data(cfg)
    # save the generated sample of data
    sample.to_csv(cfg.data.sample_path, index=False)
    # validate the initial data
    # validate_initial_data(cfg)
