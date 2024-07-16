import hydra
from omegaconf import DictConfig, OmegaConf
from src.model import fetch_features, log_metadata, train


def run(cfg: DictConfig):
    train_data_version = cfg.train_data_version

    print("Fetching features...")
    X_train, y_train = fetch_features(
        name="features_target",
        version=train_data_version,
        cfg=cfg,
    )

    print("Starting training...")
    gs = train(
        X_train,
        y_train,
        cfg,
    )

    # test_data_version = cfg.test_data_version

    print("Fetching testing features...")
    X_test, y_test = fetch_features(
        name="features_target",
        version=train_data_version,
        cfg=cfg,
        test=True,
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
    # print(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    main()
