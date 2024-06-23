"""Dataframe type for typings"""

from datetime import datetime
import hashlib
from typing_extensions import deprecated
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import FunctionTransformer

required = [
    "Year",
    "Month",
    "DayofMonth",
    # "FlightDate",
    "Cancelled",
    "OriginAirportID",
    "DepTime",
    "DepDelay",
    "DestAirportID",
    "ArrTime",
    "ArrDelay",
    "AirTime",
    "Distance",
    "ActualElapsedTime",
    "Operating_Airline",
    "Tail_Number",
]


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def pull_features(df: DataFrame) -> DataFrame:
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
    remaining_cols = set(df.columns) - set(required)

    pulled_df.drop(list(remaining_cols), axis=1, inplace=True)

    # Fix types on pulled_df
    pulled_df["Tail_Number"] = pulled_df["Tail_Number"].astype("str")

    return pulled_df


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


def hash_tail_number(df: DataFrame) -> DataFrame:
    """Hash tail numbers

    Args:
        df (DataFrame): Source dataframe

    Returns:
        Source dataframe with `Tail_Number` hashed
    """

    # Check that the column exists
    if "Tail_Number" not in df.columns:
        raise ValueError(
            "Tail_Number column is expected in the dataframe, but not found"
        )

    # Check datatype
    if df["Tail_Number"].dtype is str:
        raise ValueError("Tail_Number column's datatype is not str")

    # Hashing with buckets
    def hash_feature(text, num_buckets=1000):
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % num_buckets

    df["Tail_Number"] = df["Tail_Number"].map(hash_feature)
    return df


def process_time_data(df: DataFrame) -> DataFrame:
    """
    Process time-related data

    Args:
        df (DataFrame): Source dataframe.

    Returns:
        DataFrame: Source dataframe with time-related data transformed.
    """
    # Check that the column exists

    for c in ["DepTime", "Month", "DayofMonth"]:
        if c not in df.columns:
            raise ValueError(f"{c} is expected in the dataframe, but not found.")

    # Encoding month data
    df["Month_sin"] = sin_transformer(12).fit_transform(df["Month"])
    df["Month_cos"] = cos_transformer(12).fit_transform(df["Month"])

    # Encoding day data
    df["DayofMonth_sin"] = sin_transformer(31).fit_transform(df["DayofMonth"])
    df["DayofMonth_cos"] = cos_transformer(31).fit_transform(df["DayofMonth"])

    # Encoding hours and minutes
    df["DepTimeHH"] = df["DepTime"].apply(lambda hhmm: hhmm // 100)
    df["DepTimeMM"] = df["DepTime"].apply(lambda hhmm: hhmm % 100)

    df["DepTimeHH_sin"] = sin_transformer(24).fit_transform(df["DepTimeHH"])
    df["DepTimeHH_cos"] = cos_transformer(24).fit_transform(df["DepTimeHH"])

    df["DepTimeMM_sin"] = sin_transformer(60).fit_transform(df["DepTimeMM"])
    df["DepTimeMM_cos"] = cos_transformer(60).fit_transform(df["DepTimeMM"])

    # Drop old columns
    df.drop(
        [
            "Month",
            "DayofMonth",
            "DepTime",
            "DepTimeHH",
            "DepTimeMM",
            "Year",  # TROLLING
        ],
        axis=1,
        inplace=True,
    )

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
    