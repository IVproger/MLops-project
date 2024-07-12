import pytest
from unittest.mock import patch, MagicMock
from src.data import sample_data
from omegaconf import OmegaConf


# Fixture for a basic configuration
@pytest.fixture
def basic_cfg():
    return OmegaConf.create(
        {
            "data": {
                "url": "https://drive.google.com/uc?id=1CKXjF6xZEgjYsdwfCMxoHChI7J5WjjdI",
                "output": "datastore/data.csv",
                "datastore_path": "datastore/data.csv",
                "sample_path": "data/samples/sample.csv",
                "sample_version_path": "data/samples/sample.csv.dvc",
                "context_dir_path": "./services",
                "sample_size": 0.2,
                "last_included_row_number": 0,
                "data_version": "",
                "version_number": 0,
            }
        }
    )


# Test 1: Check if the function returns a non-None result for a valid configuration
def test_sample_data_returns_result(basic_cfg):
    result, updated_cfg = sample_data(basic_cfg)
    assert result is not None and updated_cfg is not None


# Test 2: Check if the function downloads data if not present
@patch("os.path.exists", return_value=False)
@patch("gdown.download")
def test_sample_data_downloads_data(mock_download, mock_exists, basic_cfg):
    sample_data(basic_cfg)
    mock_download.assert_called_once()


# Test 3: Check if the configuration is correctly updated
def test_sample_data_updates_configuration(basic_cfg):
    _, updated_cfg = sample_data(basic_cfg)
    assert updated_cfg.data.last_included_row_number >= 0
    assert updated_cfg.data.data_version.startswith("v")
    assert updated_cfg.data.version_number > basic_cfg.data.version_number


# Test 4: Check handling of invalid configuration
def test_sample_data_with_invalid_config():
    invalid_cfg = OmegaConf.create({"data": {}})
    result, cfg = sample_data(invalid_cfg)
    assert result is None and cfg == invalid_cfg


# Test 5: Check if start row resets when sample size exceeds total rows
@patch("builtins.open", new_callable=MagicMock)
def test_sample_data_resets_start_row(mock_open, basic_cfg):
    mock_open.return_value.__enter__.return_value = iter(["header", "row1", "row2"])
    basic_cfg.data.sample_size = 1  # 100% sample size, forcing a reset
    _, updated_cfg = sample_data(basic_cfg)
    assert updated_cfg.data.last_included_row_number < 2  # Since total_rows is 2
