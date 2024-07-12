import pytest
import pandas as pd
from unittest.mock import patch
from unittest.mock import mock_open
from omegaconf import OmegaConf
from src.data import read_datastore


# Fixture for a basic configuration
@pytest.fixture
def basic_cfg(tmp_path):
    d = tmp_path / "configs"
    d.mkdir()
    main_cfg = d / "data_sample.yaml"
    main_cfg.write_text("""
data:
  sample_path: "../sample_data.csv"
""")
    version_file = d / "data_version.txt"
    version_file.write_text("v1.0")
    return str(d)


# Test 1: Successful data reading
@patch("src.data.OmegaConf.load")
@patch("pandas.read_csv")
@patch("builtins.open", new_callable=mock_open, read_data="v1.0")
def test_successful_data_reading(
    mock_file_open, mock_read_csv, mock_load_cfg, basic_cfg
):
    mock_load_cfg.return_value = OmegaConf.load(f"{basic_cfg}/data_sample.yaml")
    mock_read_csv.return_value = pd.DataFrame({"column1": [1, 2, 3]})
    data, version = read_datastore()
    assert isinstance(data, pd.DataFrame) and version == "v1.0"


# Test 2: Missing sample path
@patch("src.data.OmegaConf.load")
def test_missing_sample_path(mock_load_cfg, basic_cfg):
    mock_load_cfg.return_value = OmegaConf.create({"data": {}})
    with pytest.raises(Exception):
        read_datastore()
