# Configuration

Tthis project we use `pydantic` to create typed configuration classes. Using `pydantic` has several advantages too.

## Configuration Sources

The main one is that we can use different source of configuration. For example, if we use the `BaseConfig` from the `config` module, we can use create something like this:

```python
from config import BaseConfig

class Config(BaseConfig):
    src: str
    dst: str

    flag: bool = False
```

### Environment Variables

The second source will be the environment variables. The name of the env variable is the same as the name in the configuration class, but in upper case. For example, the `src` attribute will be `SRC`.

```console
export SRC=/tmp/src
```

### Init Arguments

Finally, we can override the configuration using the arguments of the `Config` class. For example, we can do this:

```python
config = Config(src="/tmp/src")
```

This is very useful to create test configurations without to create a mock for the `config.ini` file or patch the environment variables.

### CLI Arguments

You can add this Configs to an `ArgumentParser` object.

```python
>>> import argparse
>>> from config import cli, BaseConfig
>>> class Config(BaseConfig):
...     src: str
...     dst: str
...     flag: bool = False
>>> parser = argparse.ArgumentParser()
>>> cli.add_model(parser, Config)
>>> args = parser.parse_args(["--src", "foo", "--dst", "bar", "--flag", "true"])
>>> Config(**vars(args))
Config(src='foo', dst="bar", flag=True)
```

## Nested Configuration Models

We can create more complex configuration for our config classes using nested models. The clear example is the `MLFlowConfig` in the `src/models/mlflow_wrappers.py` file. We can add this `pydantic` model to any of our configuration classes.

```python
from config import BaseConfig
from models.mlflow_wrappers import MLFlowConfig

class Config(BaseConfig):
    config_key: ClassVar[str] = os.getenv("CONFIG_KEY", "example")

    src: str
    dst: str

    mlflow: MLFlowConfig
```

To get the configuration from the `config.ini` we need to prefix it with our config key

```ini
[example]
    src = /tmp/src
    dst = /tmp/dst
    [example.mlflow]
        tracking_uri = http://localhost:5000
        experiment_name = test
```

And to overwrite it with env variables, we can pass a json string:

```console
export MLFLOW='{"tracking_uri": "http://localhost:5000", "experiment_name": "test"}'
```

## Credentials and Other Environment Variables

Environment variables will be picked up from the `.env` file. This file is ignored by git, so you can add your own variables there.

Typical environment variables you will need to setup:

- `DB_URL`: access to databases for reading data.
- `MLFLOW_TRACKING_URL`: access to mlflow for logging metrics and models. You can set this url to a local mlflow server for testing (<http://localhost:5000>) or even point to a local folder like `./mlruns`.
