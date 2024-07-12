import pytest
from unittest.mock import patch
from src.data import fetch_features


@patch("src.data.Client")
def test_fetch_features_returns_correct_data(mock_client):
    # Setup mock return value
    mock_client.return_value.get_artifact_version.return_value.data = (
        [1, 2, 3],
        [0, 1, 0],
    )

    # Call the function
    X, y = fetch_features()

    # Assert the expected outcome
    assert X == [1, 2, 3]
    assert y == [0, 1, 0]


@patch("src.data.Client")
def test_fetch_features_raises_exception_on_failure(mock_client):
    # Setup mock to raise an exception
    mock_client.return_value.get_artifact_version.side_effect = Exception(
        "Failed to fetch"
    )

    # Assert that an exception is raised when the function is called
    with pytest.raises(Exception) as e_info:
        fetch_features()

    assert "Failed to fetch" in str(e_info.value)
