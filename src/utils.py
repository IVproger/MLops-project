from hydra import initialize, compose


def init_hydra(config_name: str):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=config_name)
        return cfg
