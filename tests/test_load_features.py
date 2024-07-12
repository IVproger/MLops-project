import pandas as pd
import pytest
from unittest.mock import patch
from src.data import load_features


@patch("src.data.zenml.save_artifact")
def test_load_features_calls_save_artifact_with_correct_arguments(mock_save_artifact):
    # Setup
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.DataFrame({"target": [0, 1, 0]})
    version = "1.0"

    # Exercise
    load_features(X, y, version)

    # Verify
    mock_save_artifact.assert_called_once_with(
        data=[X, y], name="features_target", tags=[version]
    )


@patch("src.data.zenml.save_artifact")
def test_load_features_handles_exceptions(mock_save_artifact):
    # Setup
    mock_save_artifact.side_effect = Exception("Test exception")
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.DataFrame({"target": [0, 1, 0]})
    version = "1.0"

    # Exercise and Verify
    with pytest.raises(Exception) as exc_info:
        load_features(X, y, version)
    assert (
        str(exc_info.value) == "Test exception"
    ), "Expected load_features to raise the same exception as zenml.save_artifact"
