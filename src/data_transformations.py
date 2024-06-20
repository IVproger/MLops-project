"""Dataframe type for typings"""

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
