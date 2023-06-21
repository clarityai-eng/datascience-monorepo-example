import os
from typing import Optional, Protocol

import pandas as pd
from sklearn import datasets
from sklearn.utils import Bunch
from sqlalchemy import MetaData
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.sql import Selectable


class DataAccessProtocol(Protocol):
    engine: Engine
    metadata: MetaData

    def query(self, query: str | Selectable) -> pd.DataFrame:
        ...


class DataAccessLayer(DataAccessProtocol):
    """Provides a thread safe data access layer for any SQLAlchemy compatible storage"""

    engine: Engine
    metadata: MetaData
    conn_string: str

    def __init__(self, conn_string: Optional[str] = os.getenv("DATABASE_URL")):
        self.conn_string = conn_string

    def connect(self, conn_string: Optional[str] = None) -> "DataAccessLayer":
        """Connect to the datasource and return the connected DAL instance"""
        self.conn_string = conn_string or self.conn_string
        self.engine = create_engine(
            self.conn_string,
            echo=bool(os.getenv("ECHO_QUERY", False)),
        )
        self.metadata = MetaData(bind=self.engine)
        return self

    def query(self, query: str | Selectable) -> pd.DataFrame:
        return pd.read_sql(query, self.engine)


class InMemoryDataAccessLayer(DataAccessLayer):
    """Provides a thread safe data access layer for in memory SQLite storage"""

    engine: Engine
    metadata: MetaData
    conn_string: str

    def __init__(self, conn_string: Optional[str] = "sqlite://"):
        super().__init__(conn_string)

    def connect(self, conn_string: Optional[str] = None) -> "InMemoryDataAccessLayer":
        self.engine = create_engine(
            conn_string or self.conn_string,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self.metadata = MetaData(bind=self.engine)
        return self


class SklearnDataAccessLayer(InMemoryDataAccessLayer):
    engine: Engine
    metadata: MetaData
    conn_string: str

    def connect(self, conn_string: str | None = None) -> InMemoryDataAccessLayer:
        self.engine = create_engine(
            conn_string or self.conn_string,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self.metadata = MetaData(bind=self.engine)
        return self

    def load_data(self, dataset_name: str = "diabetes"):
        sklearn_dataset: Bunch = None
        match dataset_name:
            case "diabetes":
                sklearn_dataset = datasets.load_diabetes()
            case "iris":
                sklearn_dataset = datasets.load_iris()
            case _:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
        df["target"] = pd.Series(sklearn_dataset.target, name="target")
        df.to_sql(dataset_name, self.engine, if_exists="replace", index=False)
