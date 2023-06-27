# Data Science Monorepo

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Monorepo example for Data Science teams.

- Check out the original article [here](https://medium.com/clarityai-engineering/monorepo-in-data-science-teams-892fe64a9ef0) to understand the underlying principles of the design.
- Check out the [getting started guide](getting_started/index.md) for more information on how to use this project.

## Getting Started

### Requirements

- Python >= 3.10 (can be installed with pyenv or with asdf)
- [Pipenv](https://pipenv.pypa.io/en/latest/) == 2022.9.24
- [direnv](https://direnv.net/)
- [Docker](https://www.docker.com/)

### Setup

The project is automated using `make` and the [`Makefile`](Makefile)

In the first run, use `make setup` to bootstrap the project setup in your local. This will install dependencies using pipenv and create the necessary files. It will also run `make all` for you so you can verify that everything is working in your local.

## Project Structure

This is a ML Monorepo. Multiple modules live in the [`./src`](src/) directory. Current ones are:

```console
src
├── config             # Base configuration objects
├── data_access_layer  # Data Access layer helpers
├── datasets           # Schema for different datasets using pandera
├── feature_store      # Common interfaces for model features
└── models             # Code to generate different models
```

### Run project

Each module can implement different runs, but in general they should be runable as a module. For eample:

```console
python -m models.diabetes.features
```

When running the project as a docker image, you must specify the module to run as the docker command. We have implemented a helper script to run the docker build image with the latest project files and AWS credentials setup

```console
./scripts/docker-run models.diabetes.features
```

## End-to-end example: Diabetes prediction

1. Install the project dependencies in the pipenv environment.
1. Run mlflow in local with model registry:

    ```console
    $ pipenv run mlflow
    [INFO] Starting gunicorn 20.1.0
    [INFO] Listening at: http://127.0.0.1:5000
    ```

1. Run the feature extraction:

    ```console
    $ pipenv run python -m models.diabetes.features \
        --dst tmp/data/diabetes_features.parquet
    features.py:main:14 INFO: Start | run
    features.py:main:14 INFO: End | run | (result=sklearn_dataset='diabetes' dst='tmp/data/diabetes_features.parquet')
    ```

1. Run the data preprocessing:

    ```console
    $ pipenv run python -m models.preprocess \
        --src_features=tmp/data/diabetes_features.parquet \
        --dst_x_train=tmp/data/x_train.parquet \
        --dst_y_train=tmp/data/y_train.parquet \
        --dst_x_test=tmp/data/x_test.parquet \
        --dst_y_test=tmp/data/y_test.parquet
    preprocess.py:<module>:46 INFO: Start | run
    preprocess.py:<module>:46 INFO: End | run | (result=src_features='tmp/data/diabetes_features.parquet' dst_x_train='tmp/data/x_train.parquet' dst_y_train='tmp/data/y_train.parquet' dst_x_test='tmp/data/x_test.parquet' dst_y_test='tmp/data/y_test.parquet')
    ```

1. Run the model training (note the output model uri)

    ```console
    $ pipenv run python -m models.diabetes.train \
        --src_x_train=tmp/data/x_train.parquet \
        --src_y_train=tmp/data/y_train.parquet \
        --src_x_test=tmp/data/x_test.parquet \
        --src_y_test=tmp/data/y_test.parquet
    train.py:<module>:72 INFO: Start | run
    train.py:<module>:72 INFO: End | run
    train.py:<module>:75 INFO: Model saved to runs:/2099249145894ae3b16b7a37653cec06/model
    ```

1. Run a predictions using the previous logged model in mlflow

    ```console
    $ pipenv run python -m models.predict \
        --src_features=tmp/data/x_test.parquet \
        --src_model=runs:/2099249145894ae3b16b7a37653cec06/model \
        --dst_y_hat=tmp/data/y_hat_test.parquet
    predict.py:<module>:126 INFO: src_features='tmp/data/x_test.parquet' src_model='runs:/2099249145894ae3b16b7a37653cec06/model' flavour='sklearn' parallel_backend='threading' n_jobs=-1 batch_predictions=False batch_size=10000 progress_bar=True dst_y_hat='tmp/data/y_hat_test.parquet' mlflow=MLFlowConfig(experiment_name='predict', run_name='run-2023-06-20T18-09-48', tracking_uri=SecretStr('**********'), flavor='sklearn', tags=None) execution_date='2023-06-20T18:09:48Z'
    predict.py:<module>:131 INFO: Reading data from tmp/data/x_test.parquet
    predict.py:<module>:133 INFO: Loading model from runs:/2099249145894ae3b16b7a37653cec06/model
    predict.py:<module>:137 INFO: Writing y_hat to tmp/data/y_hat_test.parquet
    ```

## Documentation

Documentation uses `mkdocs`.

To run the documentation locally, run `make mkdocs` and open <http://localhost:8000>

To expand the documentation, edit the files in the [`./docs`](docs/) directory. Any markdown file can be added there and will be rendered in the documentation with navigation and search support.
