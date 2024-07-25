import pandas as pd

from src.data import preprocess_data
from src.utils import init_hydra


def test_preprocess_data():
    cfg = init_hydra("main")

    df = pd.read_csv("tests/samples/test.csv")
    df_transformed = pd.read_csv("tests/samples/test_transformed.csv")

    fresh_df = pd.concat([*preprocess_data(cfg, df)], axis=1).round(6)

    assert (
        df_transformed.compare(fresh_df).size == 0
    ), "Expected correct transformations"
