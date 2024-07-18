import warnings

warnings.filterwarnings("ignore")

from src.data import extract_data
from src.utils import init_hydra
from src.data import preprocess_data
import giskard


def wrap_dataset():
    """
    Wrap raw dataset
    :return: giskard_dataset, version
    """
    cfg = init_hydra("main")

    df, version = extract_data(version=cfg.test_data_version, cfg=cfg)

    dataset_name = cfg.data.dataset_name
    TARGET_COLUMN = cfg.data.target_cols[0]

    giskard_dataset = giskard.Dataset(
        df=df,
        target=TARGET_COLUMN,
        name=dataset_name,
    )

    return giskard_dataset, version


# TODO
def wrap_model():
    """
    Fetch model from model registry
    :return: model
    """
    pass


def predict(model, raw_df):
    """
    Define custom predict function
    """
    X, _ = preprocess_data(df=raw_df)
    return model.predict(X)


if __name__ == "__main__":
    pass
