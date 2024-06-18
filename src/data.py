'''
Import the necessary libraries and modules for the data sampling script.
'''
import os
import gdown
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def sample_data(cfg: DictConfig) -> None:
    '''
    The function to sample the data from the given URL and save it to the sample path.
    '''
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

if __name__ == "__main__":
    cfg = OmegaConf.load("../configs/main.yaml")
    sample = sample_data(cfg)
    # save the generated sample of data
    sample.to_csv(cfg.data.sample_path, index=False)
    