"""Dataframe type for typings"""

from datetime import datetime
from pandas import DataFrame

required = [
    "FlightDate",
    "Cancelled",
    "DepTime",
    "DepDelay",
    "ArrTime",
    "AirTime",
    "ActualElapsedTime",
    "Distance",
    "Operating_Airline",
    "Tail_Number",
    "OriginAirportID",
    "DestAirportID",
    "ArrDelay",
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

    return pulled_df.drop(list(remaining_cols), axis=1)


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

    df["FlightDate"] = df["FlightDate"].map(
        lambda d: datetime.strptime(d, "%Y-%m-%d")
    )
    return df
