"""Dataframe type for typings"""

import hashlib
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FunctionTransformer

from hydra import compose, initialize
from deprecation import deprecated


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


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


def encode_cyclic_time_data(df: DataFrame, col: str, period: int) -> DataFrame:
    # Check that the column exists
    if col not in df.columns:
        raise ValueError(f"{col} is expected in the dataframe, but not found.")

    # Encode data
    df[col + "_sin"] = sin_transformer(period).fit_transform(df[col])
    df[col + "_cos"] = cos_transformer(period).fit_transform(df[col])

    df.drop([col], axis=1, inplace=True)

    return df


def fix_hhmm(df: DataFrame, col: str) -> tuple[DataFrame, str, str]:
    # # Encoding hours and minutes
    colHH = col + "HH"
    colMM = col + "MM"
    df[colHH] = df[col].apply(lambda hhmm: hhmm // 100)
    df[colMM] = df[col].apply(lambda hhmm: hhmm % 100)

    df.drop([col], axis=1, inplace=True)
    return (df, colHH, colMM)


@deprecated
def fix_dtypes(df: DataFrame) -> DataFrame:
    """Fixes datatypes for numerical columns

    Args:
        df (DataFrame): Source dataframe

    Returns:
        DataFrame: Source dataframe with types fixed
    """
    for c in [
        "DepTime",
        "DepDelay",
        "ArrTime",
        "AirTime",
        "ActualElapsedTime",
        "Distance",
        "OriginAirportID",
        "DestAirportID",
        "ArrDelay",
    ]:
        df[c] = df[c].astype("int64")
    return df


@deprecated
def encode_op_airline(df: DataFrame) -> DataFrame:
    """Encode `Operating_Airline` with onehot encoding

    Args:
        df (DataFrame): Source dataframe

    Returns:
        Source dataframe with `Operating_Airline` onehotencoded
    """
    # Check that the column exists
    if "Operating_Airline" not in df.columns:
        raise ValueError(
            "Operating_Airline column is expected in the dataframe, but not found"
        )

    # Check datatype
    if df["Operating_Airline"].dtype is str:
        raise ValueError("Operating_Airline column's datatype is not str")

    df = pd.get_dummies(df, columns=["Operating_Airline"])
    return df


def handle_missing_values(df: DataFrame) -> DataFrame:
    """Handle missing values in the dataframe

    Args:
        df (DataFrame): Source dataframe

    Returns:
        Source dataframe with missing values handled
    """
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        # Fill missing values with 0
        df.fillna(0, inplace=True)

    return df


def handle_duplicates(df: DataFrame) -> DataFrame:
    """Handle duplicates in the dataframe

    Args:
        df (DataFrame): Source dataframe

    Returns:
        Source dataframe with duplicates handled
    """
    # Check for duplicates
    if df.duplicated().sum() > 0:
        # Drop duplicates
        df.drop_duplicates(inplace=True)

    return df


def scale(df: DataFrame) -> DataFrame:
    """Scale the dataframe using sklearn's StandardScaler

    Args:
        df (DataFrame): Source dataframe

    Returns:
        DataFrame: Source dataframe scaled
    """
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df
