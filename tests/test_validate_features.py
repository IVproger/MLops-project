import pandas as pd
import pytest
from src.data import validate_features


def test_validate_features():
    df = pd.read_csv("tests/samples/test_transformed.csv")
    X = df.drop(["Cancelled"], axis=1)
    y = df["Cancelled"]

    assert validate_features(X, y)


def test_validate_features_fail():
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.DataFrame({"target": [0, 1, 0]})

    with pytest.raises(Exception) as excInfo:
        validate_features(X, y)
    assert "Validation failed" in str(excInfo.value)
