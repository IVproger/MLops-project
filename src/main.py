import hydra
from omegaconf import DictConfig
from src.model import fetch_features, log_metadata, train  # noqa: E402


def run(cfg: DictConfig):
    train_data_version = cfg.train_data_version
    name = cfg.data.artifact_name

    print("Fetching features...")
    X_train, y_train = fetch_features(
        name=name,
        version=train_data_version,
        cfg=cfg,
    )

    print("Starting training...")
    gs = train(
        X_train,
        y_train,
        cfg,
    )

    test_data_version = cfg.test_data_version
    X_test, y_test = fetch_features(
        name="features_target",
        version=test_data_version,
        cfg=cfg,
    )

    print("Logging metadata...")
    log_metadata(
        cfg,
        gs,
        X_train,
        y_train,
        X_test,
        y_test,
    )


@hydra.main(config_path="../configs", config_name="main", version_base=None)  # type: ignore
def main(cfg=None):
    run(cfg)


if __name__ == "__main__":
    main()
