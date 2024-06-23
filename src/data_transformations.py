"""Dataframe type for typings"""

from datetime import datetime
import hashlib
from typing_extensions import deprecated
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import Normalizer

required = [
    "Year",
    "Month",
    "DayofMonth",
    "Year",
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


@deprecated
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


@deprecated
def sync_times(df: DataFrame) -> DataFrame:
    """
    Transform `DepTime` & `AirTime` columns to minutes.

    Args:
        df (DataFrame): Source dataframe.

    Returns:
        DataFrame: Source dataframe with `DepTime` & `AirTime` columns' time transformed to minutes.
    """
    # Check that the column exists
    for c in ["DepTime", "AirTime"]:
        if c not in df.columns:
            raise ValueError(f"{c} column is expected in the dataframe, but not found.")

    # Check datatype and handle non-numeric values gracefully
    for c in ["DepTime", "AirTime"]:
        if (
            not df[c].dtype.kind in "biufc"
        ):  # Checks if the data type is numeric or complex
            raise ValueError(f"`{c}` datatype is not numeric.")

    def hhmm2minutes(raw):
        if pd.isna(raw):  # More efficient NaN check
            return raw
        hhmm = int(raw)
        strhhmm = str(hhmm).zfill(4)
        hour = int(strhhmm[:2])
        minutes = int(strhhmm[2:])
        return hour * 60 + minutes

    for c in ["DepTime", "AirTime"]:
        df[c] = df[c].apply(hhmm2minutes)  # Using apply for potential NaN handling

    return df


@deprecated
def convert_to_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert `FlightDate` column to seconds since epoch (January 1, 1970).

    Args:
        df (pd.DataFrame): Source dataframe with a `FlightDate` column.

    Returns:
        pd.DataFrame: DataFrame with `FlightDate` column converted to epoch time.
    """
    # Ensure FlightDate is in datetime format
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    # Convert FlightDate to epoch time
    df["FlightDate"] = (df["FlightDate"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(
        "1s"
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


@deprecated
def normalize(df: DataFrame) -> DataFrame:
    """Normalize the dataframe using sklearn's Normalizer

    Args:
        df (DataFrame): Source dataframe

    Returns:
        DataFrame: Source dataframe normalized
    """
    # Initialize the Normalizer
    normalizer = Normalizer()

    # Select numeric columns for normalization
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    # Normalize the numeric columns
    df[numeric_cols] = normalizer.fit_transform(df[numeric_cols])

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
    