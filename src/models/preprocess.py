"""Example module to do preprocessing before training."""
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from config import BaseConfig, cli
from config.logging import logger_wraps


class Config(BaseConfig):
    src_features: str

    dst_x_train: str
    dst_y_train: str

    dst_x_test: str
    dst_y_test: str


class Preprocess:
    def __init__(self, config: Config) -> None:
        self.config = config

    @logger_wraps(outputs=True)
    def run(self):
        data = pd.read_parquet(self.config.src_features)
        X = data.drop(columns=["target"])
        y = data["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        X_train.to_parquet(self.config.dst_x_train)
        X_test.to_parquet(self.config.dst_x_test)

        y_train.to_frame(name="target").to_parquet(self.config.dst_y_train)
        y_test.to_frame(name="target").to_parquet(self.config.dst_y_test)
        return self.config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cli.add_model(parser, Config)
    args = parser.parse_args()
    config = Config(**vars(args))
    Preprocess(config).run()
