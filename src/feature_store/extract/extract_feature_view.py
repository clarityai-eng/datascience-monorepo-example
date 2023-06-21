from typing import Dict, Type

from pydantic import PyObject, SecretStr

from config import BaseConfig
from config.logging import logger_wraps
from data_access_layer import DataAccessLayer
from data_access_layer.files import create_dir_if_not_exists
from feature_store.feature_views import BaseFeatureView


class Config(BaseConfig):
    feature_view: PyObject = Type[BaseFeatureView]
    db_url: SecretStr
    read_kwargs: Dict = {}

    dst: str


class ExtractTask:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.dal = DataAccessLayer(conn_string=config.db_url.get_secret_value()).connect()

    @logger_wraps(outputs=True)
    def run(self):
        data = self.config.feature_view.read(self.dal, **self.config.read_kwargs)
        create_dir_if_not_exists(self.config.dst)
        data.to_parquet(self.config.dst)
        return self.config.dst


if __name__ == "__main__":
    config = Config()
    task = ExtractTask(config)
    task.run()
