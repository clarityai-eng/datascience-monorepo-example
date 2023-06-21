from config import BaseConfig
from config.logging import logger_wraps
from data_access_layer.dal import SklearnDataAccessLayer
from data_access_layer.files import create_dir_if_not_exists
from feature_store.feature_views import DiabetesFeatureView, IrisFeatureView


class Config(BaseConfig):
    sklearn_dataset: str = "diabetes"
    dst: str


class ExtractTask:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.dal = SklearnDataAccessLayer().connect()

    @logger_wraps(outputs=True)
    def run(self):
        match self.config.sklearn_dataset:
            case "diabetes":
                data = DiabetesFeatureView.read(self.dal)
            case "iris":
                data = IrisFeatureView.read(self.dal)
            case _:
                raise ValueError(f"Unknown sklearn dataset: {self.config.sklearn_dataset}")

        create_dir_if_not_exists(self.config.dst)
        data.to_parquet(self.config.dst)
        return self.config


if __name__ == "__main__":
    config = Config()
    task = ExtractTask(config)
    task.run()
