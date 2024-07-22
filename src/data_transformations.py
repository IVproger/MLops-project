"""Dataframe type for typings"""

import hashlib
import numpy as np
from pandas import DataFrame
from sklearn.pipeline import FunctionTransformer


def pull_features(df: DataFrame, required: list[str]) -> DataFrame:
    """
    Extract only the required features from the dataframe
    """
    # Check that the required columns are there
    for c in required:
        if c not in df.columns:
            raise ValueError(
                f"Dataframe lacks one or more of the required columns: {c}"
            )
    pulled_df = df.copy()
    columns_to_drop = set(df.columns) - set(required)

    pulled_df.drop(list(columns_to_drop), axis=1, inplace=True)

    return pulled_df


def hash_feature(df: DataFrame, col: str, num_buckets=1000):
    # Hashing with buckets
    df[col] = df[col].map(
        lambda text: int(hashlib.md5(text.encode()).hexdigest(), 16) % num_buckets
    )
    return df


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def encode_cyclic_time_data(df: DataFrame, col: str, period: int) -> DataFrame:
    # Check that the column exists
    if col not in df.columns:
        raise ValueError(f"{col} is expected in the dataframe, but not found.")

    # Encode data
    df[col + "_sin"] = sin_transformer(period).fit_transform(df[col])
    df[col + "_cos"] = cos_transformer(period).fit_transform(df[col])

    # df.drop([col], axis=1, inplace=True)

    return df


def fix_hhmm(df: DataFrame, col: str) -> tuple[DataFrame, str, str]:
    # Encoding hours and minutes
    colHH = col + "HH"
    colMM = col + "MM"
    df[colHH] = df[col].apply(lambda hhmm: hhmm // 100)
    df[colMM] = df[col].apply(lambda hhmm: hhmm % 100)

    df.drop([col], axis=1, inplace=True)
    return (df, colHH, colMM)
