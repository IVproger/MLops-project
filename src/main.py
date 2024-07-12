import hydra
from src.model import fetch_features


def run(args):
    cfg = args

    train_data_version = cfg.train_data_version

    X_train, y_train = fetch_features(
        name="features_target", version=train_data_version
    )

    print(X_train, y_train)


@hydra.main(config_path="../configs", config_name="main", version_base=None)  # type: ignore
def main(cfg=None):
    # print(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    main()
