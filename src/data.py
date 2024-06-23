"""
Import the necessary libraries and modules for the data sampling script.
"""
import os
import gdown
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from src.data_quality import load_context_and_sample_data, define_expectations

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def sample_data(cfg: DictConfig):
    """
    The function to sample the data from the given URL and save it to the sample path.
    """
    try:
        datastore_path = cfg.data.datastore_path
        sample_path = cfg.data.sample_path
        # Check if the source data is available, if not download it.
        if not os.path.exists(datastore_path):
            print("Downloading data from: ", cfg.data.url)
            gdown.download(cfg.data.url, cfg.data.output, quiet=False)
        # Read the data from the source 
        data = pd.read_csv(datastore_path)
        # We should be sure that every time when we take the sample,
        # it should be different from the previous one.
        if os.path.exists(sample_path):
            print("Previous sample found, removing it from the data.")
            # drop the previous sample from the data to avoid duplication
            sample_tmp = pd.read_csv(sample_path)
            data = data.drop(sample_tmp.index)
        # Data sampling process
        print("Sampling data...")
        resulted_sample = data.sample(frac=cfg.data.sample_size)
        return resulted_sample
    except Exception as e:
        print("Error in loading or sampling the data: ", e)
        return None


@hydra.main(version_base=None, config_path="../configs", config_name="main")
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


if __name__ == "__main__":
    cfg = OmegaConf.load("../configs/main.yaml")
    sample = sample_data(cfg)
    # save the generated sample of data
    sample.to_csv(cfg.data.sample_path, index=False)
    # validate the initial data
    validate_initial_data(cfg)
