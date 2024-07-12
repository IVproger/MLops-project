import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
import pandas as pd
from src.data import validate_initial_data


# Fixture for a basic configuration
@pytest.fixture
def basic_cfg():
    return OmegaConf.create(
        {
            "data": {
                "context_dir_path": "./services",
            }
        }
    )


# Fixture for a valid DataFrame
@pytest.fixture
def valid_df():
    data = pd.read_csv("data/samples/sample.csv")
    return data


# Test 1: Successful validation
@patch("src.data.gx.get_context")
def test_successful_validation(mock_get_context, basic_cfg, valid_df):
    mock_context = MagicMock()
    mock_checkpoint = MagicMock()
    mock_checkpoint_result = MagicMock(success=True)
    mock_checkpoint.run.return_value = mock_checkpoint_result
    mock_context.get_checkpoint.return_value = mock_checkpoint
    mock_get_context.return_value = mock_context
    assert validate_initial_data(basic_cfg, valid_df) is True


# Test 2: Failed validation
@patch("src.data.gx.get_context")
def test_failed_validation(mock_get_context, basic_cfg, valid_df):
    mock_context = MagicMock()
    mock_checkpoint = MagicMock()
    mock_checkpoint_result = MagicMock(success=False)
    mock_checkpoint.run.return_value = mock_checkpoint_result
    mock_context.get_checkpoint.return_value = mock_checkpoint
    mock_get_context.return_value = mock_context
    assert validate_initial_data(basic_cfg, valid_df) is False


# Test 3: Invalid configuration
def test_with_invalid_config(valid_df):
    invalid_cfg = OmegaConf.create({"data": {}})
    assert validate_initial_data(invalid_cfg, valid_df) is False


# Test 4: Exception handling
@patch("src.data.gx.get_context")
def test_exception_handling(mock_get_context, basic_cfg, valid_df):
    mock_get_context.side_effect = Exception("Test exception")
    assert validate_initial_data(basic_cfg, valid_df) is False


# Test 5: Valid configuration but invalid DataFrame
@patch("src.data.gx.get_context")
def test_invalid_dataframe(mock_get_context, basic_cfg):
    invalid_df = pd.DataFrame({"invalid_column": [1, 2, 3]})
    mock_context = MagicMock()
    mock_checkpoint = MagicMock()
    mock_checkpoint_result = MagicMock(
        success=False
    )  # Assuming validation fails for this DataFrame
    mock_checkpoint.run.return_value = mock_checkpoint_result
    mock_context.get_checkpoint.return_value = mock_checkpoint
    mock_get_context.return_value = mock_context
    assert validate_initial_data(basic_cfg, invalid_df) is False
