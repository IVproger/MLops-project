import pandas as pd
import pytest
from src.data import validate_initial_data
from src.utils import init_hydra


def test_validate_initial_data():
    cfg = init_hydra("main")
    df = pd.read_csv("tests/samples/test.csv")

    try:
        validate_initial_data(cfg, df)
    except ValueError:
        pytest.fail("Validation failed")


def test_validate_initial_data_fail():
    cfg = init_hydra("main")
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0],
        }
    )

    with pytest.raises(ValueError) as excInfo:
        validate_initial_data(cfg, df)
    assert "Validation failed" in str(excInfo.value)
