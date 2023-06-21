from typing import Any, Dict

import pandas as pd
import pandera as pa
from loguru import logger
from pandera.typing import Series
from pandera.typing.pandas import DataFrame

from datasets import options


class BaseSchema(pa.DataFrameModel):
    """Base schema with helpers and validation functions checks."""

    class Config:
        coerce = True
        strict = "filter"  # NOTE: remove columns not specified

    @pa.check(r"[\S]*_date[\S]*", regex=True)
    def date_time_iso_format(cls, date_series: Series[int]) -> Series[bool]:
        """Check any date column is a iso string"""
        return Series(date_series.str.fullmatch(options.DATE_ISO_REGEX))

    @classmethod
    def fill_missing_cols(cls, data: pd.DataFrame | DataFrame) -> pd.DataFrame | DataFrame:
        columns = list(cls.to_schema().columns)
        cols_to_fill = [c for c in columns if c not in data]
        if cols_to_fill:
            logger.warning(
                "Data for table {table} is missing columns: {columns}. Filling with None",
                table=cls.__class__,
                columns=cols_to_fill,
            )
            for c in cols_to_fill:
                data[c] = None
        return data

    @classmethod
    def coerce_filter_columns(cls, data: DataFrame) -> DataFrame["BaseSchema"]:
        """Apply schema to data and filter out columns that are not in the schema.

        This is a more performant alternative to `schema.validate(data)`. However, it
        does not raise an error if the data does not conform to the schema, and it
        won't apply any pandera custom checks.
        """
        schema = cls.to_schema()
        schema_dtypes: Dict[str, Any] = {c: pandera_type.type for c, pandera_type in schema.get_dtypes(data).items()}
        columns = list(schema_dtypes.keys())
        return data[columns].astype(schema_dtypes)
