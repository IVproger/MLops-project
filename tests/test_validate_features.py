import pandas as pd
from src.data import validate_features


def test_validate_features_returns_dataframes():
    # Setup
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.DataFrame({"target": [0, 1, 0]})

    # Exercise
    validated_X, validated_y = validate_features(X, y)

    # Verify
    assert isinstance(
        validated_X, pd.DataFrame
    ), "Expected validated_X to be a pd.DataFrame"
    assert isinstance(
        validated_y, pd.DataFrame
    ), "Expected validated_y to be a pd.DataFrame"


def test_validate_features_preserves_shape():
    # Setup
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.DataFrame({"target": [0, 1, 0]})

    # Exercise
    validated_X, validated_y = validate_features(X, y)

    # Verify
    assert (
        validated_X.shape == X.shape
    ), "Expected validated_X to have the same shape as input X"
    assert (
        validated_y.shape == y.shape
    ), "Expected validated_y to have the same shape as input y"
