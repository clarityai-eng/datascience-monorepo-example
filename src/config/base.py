from loguru import logger
from pydantic import BaseModel, BaseSettings, Extra, validator


class BaseModelValidator(BaseModel):
    @validator("*", pre=True, allow_reuse=True)
    def validate_complex_params(cls, v, config, field):
        if isinstance(v, str) and field.is_complex() and isinstance(v, str):
            try:
                return config.json_loads(v)  # type: ignore
            except ValueError as e:
                logger.error(f"Complex model field type {field.name} cannot be parsed from json string: {v}")
                raise e
        return v

    class Config:
        allow_mutation = False


class BaseConfig(BaseModelValidator, BaseSettings):
    """Base config class for type safe config management"""

    class Config:
        env_file_encoding = "utf-8"
        extra = Extra.forbid
