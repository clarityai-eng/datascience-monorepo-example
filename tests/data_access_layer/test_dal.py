from abc import ABCMeta, abstractmethod

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from testcontainers.postgres import PostgresContainer

from data_access_layer import DataAccessLayer, InMemoryDataAccessLayer


@pytest.fixture(scope="session")
def db_url() -> str:
    return "sqlite://"


class BaseDALTester(metaclass=ABCMeta):
    @abstractmethod
    @pytest.fixture(scope="class")
    def dal(self, db_url):
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def table_name(self) -> str:
        return "testing_data"

    @pytest.fixture(scope="class")
    def mock_data(self, dal: DataAccessLayer, table_name: str):
        data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [20, 30, 40],
            }
        )
        data.to_sql(table_name, con=dal.engine, if_exists="replace", index=False)
        yield data
        with dal.engine.connect() as con:
            con.execute(f"DROP TABLE {table_name};")

    def test_get_data_from_dal(self, table_name: str, dal: DataAccessLayer, mock_data: pd.DataFrame):
        result = pd.read_sql_table(table_name=table_name, con=dal.engine)
        expected = mock_data

        assert_frame_equal(result, expected, check_like=True)  # NOTE: ignore column order


class TestDAL(BaseDALTester):
    @pytest.fixture(scope="class")
    def dal(self):
        with PostgresContainer("public.ecr.aws/docker/library/postgres:13.2") as postgres:
            db_url = postgres.get_connection_url()
            dal = DataAccessLayer(db_url).connect()
            yield dal
            dal.metadata.drop_all()


class TestDALInMemory(BaseDALTester):
    @pytest.fixture(scope="class")
    def dal(self, db_url):
        dal = InMemoryDataAccessLayer(db_url).connect()
        yield dal
        dal.metadata.drop_all()
