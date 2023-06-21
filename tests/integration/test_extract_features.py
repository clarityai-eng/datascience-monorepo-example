from pathlib import Path

import pytest

from feature_store.extract import extract_sklearn_feature_view


@pytest.fixture(scope="session")
def db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    tmp_path = tmp_path_factory.mktemp("data").joinpath("data.db")
    tmp_path.touch()
    uri = f"sqlite:///{tmp_path}"
    return uri


class TestExtractSklearnFeatureView:
    @pytest.fixture
    def config(self, tmp_path_factory: pytest.TempPathFactory) -> extract_sklearn_feature_view.Config:
        data_dir = tmp_path_factory.mktemp("data")
        config = extract_sklearn_feature_view.Config(
            sklearn_dataset="diabetes",
            dst=str(data_dir.joinpath("result.parquet")),
        )
        return config

    def test_run(self, config: extract_sklearn_feature_view.Config):
        task = extract_sklearn_feature_view.ExtractTask(config)
        result = task.run()
        assert result.dst == config.dst
        assert Path(result.dst).exists()
