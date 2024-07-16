from typing import Annotated

import pandas as pd
from zenml import pipeline, step, ArtifactConfig

from src.data import read_datastore, preprocess_data, validate_features, load_features


@step(enable_cache=False)
def extract() -> (
    tuple[
        Annotated[
            pd.DataFrame,
            ArtifactConfig(name="extracted_data", tags=["data_preparation"]),
        ],
        Annotated[str, ArtifactConfig(name="data_version", tags=["data_preparation"])],
    ]
):
    """
    Extract the sample data from the data store, and its version.
    """
    df, version = read_datastore()
    print("Extracted from datastore", version)
    return df, version


@step(enable_cache=False)
def transform(
    df: pd.DataFrame,
) -> tuple[
    Annotated[
        pd.DataFrame, ArtifactConfig(name="input_features", tags=["data_preparation"])
    ],
    Annotated[
        pd.DataFrame, ArtifactConfig(name="input_target", tags=["data_preparation"])
    ],
]:
    """
    Transform the input data to extract features and target.
    """
    from hydra import initialize, compose
    
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="data_transformations")

        X, y = preprocess_data(cfg, df)
        return X, y
    

@step(enable_cache=False)
def validate(
    X: pd.DataFrame, y: pd.DataFrame
) -> tuple[
    Annotated[
        pd.DataFrame,
        ArtifactConfig(name="valid_input_features", tags=["data_preparation"]),
    ],
    Annotated[
        pd.DataFrame, ArtifactConfig(name="valid_target", tags=["data_preparation"])
    ],
]:
    """
    Validate the features and target using Great Expectations.
    """
    X, y = validate_features(X, y)
    return X, y


@step(enable_cache=False)
def load(
    X: pd.DataFrame, y: pd.DataFrame, version: str
) -> tuple[
    Annotated[pd.DataFrame, ArtifactConfig(name="features", tags=["data_preparation"])],
    Annotated[pd.DataFrame, ArtifactConfig(name="target", tags=["data_preparation"])],
]:
    """
    Save the features and target as artifact to ZenML.
    """
    load_features(X, y, version)
    print("Loaded into features store", version)
    return X, y


@pipeline()
def prepare_data_pipeline():
    df, version = extract()
    X, y = transform(df)
    X, y = validate(X, y)
    X, y = load(X, y, version)


if __name__ == "__main__":
    run = prepare_data_pipeline()
