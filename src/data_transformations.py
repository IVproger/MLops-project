"""Dataframe type for typings"""

from datetime import datetime
import hashlib
import pandas as pd
from pandas import DataFrame

required = [
    "FlightDate",
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


def pull_features(df: DataFrame):
    """
    Extract only the required features from the dataframe
    """
    # Check that the required columns are there
    for c in required:
        print(c)
        if c not in df.columns:
            raise ValueError(
                f"Dataframe lacks one or more of the required columns: {c}"
            )
    pulled_df = df.copy()
    remaining_cols = set(df.columns) - set(required)

    pulled_df.drop(list(remaining_cols), axis=1, inplace=True)

    # Fix types
    for c in [
        "DepTime",
        "DepDelay",
        "ArrTime",
        "ArrDelay",
    ]:
        df[c] = df[c].astype("int64")


def str2date(df: DataFrame) -> DataFrame:
    """Transform FlightDate column from str to date.
    Transformations occur in-place

    Args:
        df (DataFrame): Source dataframe

    Returns:
        DataFrame: Source dataframe with FlightDate mapped to datetime datatype
    """
    # Check that the column exists
    if "FlightDate" not in df.columns:
        raise ValueError(
            "FlightDate column is expected in the dataframe, but not found"
        )

    # Check datatype
    if df["FlightDate"].dtype is str:
        raise ValueError("FlightDate column's datatype is not str")

    df["FlightDate"] = df["FlightDate"].map(lambda d: datetime.strptime(d, "%Y-%m-%d"))
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


def sync_times(df: DataFrame) -> DataFrame:
    """
    Transform `DepTime` & `AirTime` columns to minutes

    Args:
        df (DataFrame): Source dataframe

    Returns:
        Source dataframe with `DepTime` & `AirTime` columns' time transformed to minutes
    """
    # Check that the column exists
    if ["DepTime", "AirTime"] not in df.columns:
        raise ValueError(
            "[DepTime, AirTime] columns are expected in the dataframe, but not found"
        )

    # Check datatype
    for c in ["DepTime", "AirTime"]:
        if df[c].dtype is not "int64":
            raise ValueError(f"`{c}` datatype is not `int64`")

    def hhmm2minutes(hhmm: int):
        strhhmm = str(hhmm).zfill(4)
        hour = int(strhhmm[:2])
        minutes = int(strhhmm[2:])

        return hour * 60 + minutes

    for c in ["DepTime", "AirTime"]:
        df[c] = df[c].map(hhmm2minutes)

    return df
