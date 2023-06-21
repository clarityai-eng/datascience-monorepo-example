import pandas as pd
import pandera as pa

from data_access_layer.dal import DataAccessLayer, SklearnDataAccessLayer
from datasets.datasets import BaseSchema


class BaseFeatureView(BaseSchema):
    class Config:
        coerce = True
        strict = "filter"

    @classmethod
    def read(cls, dal: DataAccessLayer, *, fast_coerce_types: bool = False, **kwargs):
        raise NotImplementedError


class DiabetesFeatureView(BaseFeatureView):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    target: float

    @classmethod
    def read(cls, dal: SklearnDataAccessLayer, **kwargs):
        dal.load_data("diabetes")
        data = pd.read_sql_table("diabetes", dal.engine)
        return cls(data)


class IrisFeatureView(BaseFeatureView):
    sepal_length: float = pa.Field(alias="sepal length (cm)")
    sepal_width: float = pa.Field(alias="sepal width (cm)")
    petal_length: float = pa.Field(alias="petal length (cm)")
    petal_width: float = pa.Field(alias="petal width (cm)")
    target: int

    @classmethod
    def read(cls, dal: SklearnDataAccessLayer, **kwargs):
        dal.load_data()
        data = pd.read_sql_table("diabetes", dal.engine)
        return cls(data)
