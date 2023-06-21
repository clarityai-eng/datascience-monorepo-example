# MLFlow Usage

[MLFlow](https://mlflow.org/docs/latest/index.html) is the our tool for experiment tracking and model versioning.

We have implemented some wrappers around MLFlow, but it is encouraged to always use the MLFlow native API, with the [client](https://mlflow.org/docs/latest/python_api/mlflow.client.html#module-mlflow.client) or the [fluent API](https://mlflow.org/docs/latest/python_api/mlflow.html)

## Local vs Remote

MLFlow has different deployment options that you can check [here](https://mlflow.org/docs/latest/tracking.html#how-runs-and-artifacts-are-recorded). For our purposes we can distinguish 3 scenarios:

### Local Without Model Registry

This is the simplest one. It will record the experiments inside a folder. The default folder name is `./mlruns`. This is the default behavior if you start using MLFlow out of the box.

```python
import mlflow

mlflow.set_tracking_uri("./mlruns")
```

To launch the server with this configuration, you can run the following command:

```console
mlflow server
```

And access <http://localhost:5000>.

### Local With Model Registry **[Recommended]**

The problem with the above setup is that you won't have access to the model registry functionality. To fix this, we need to use a local database system. For this, we have two approaches.

1. Use the pipenv script to launch mlflow for your local environment. This will store the metadata in a sqlite db (`./mlruns.db`) and the artifacts in your local file storage as the previous section (`./mlruns`)

    ```console
    ❯ pipenv run mlflow
    ❯ # The following command is the equivalent
    ❯ mlflow server --backend-store-uri=sqlite:///mlruns.db --default-artifact-root=file:mlruns
    ```

2. **[Recommended]** Use the docker-compose local stack. This option will store the metadata in a postgresql container and the artifacts in a S3 compatible server named [minio](https://min.io/). The artifact storage is [proxied](https://www.mlflow.org/docs/latest/tracking.html#scenario-5-mlflow-tracking-server-enabled-with-proxied-artifact-storage-access), which means that the client setup is easier as we only need to connect to the mlflow server. Otherwise, the client would need permissions to access the S3 minio storage.

    ```console
    ❯ make up  # Launch the stack. Alias to docker compose up -d
    [+] Running 7/7
    ✔ Network datascience-monorepo-example_default           Created         0.0s
    ✔ Volume "datascience-monorepo-example_mlflow-minio"     Created         0.0s
    ✔ Volume "datascience-monorepo-example_mlflow-postgres"  Created         0.0s
    ✔ Container mlflow-db                         Healthy         7.4s
    ✔ Container mlflow-minio-s3                   Healthy         4.4s
    ✔ Container mlflow-create-bucket              Exited          4.4s
    ✔ Container mlflow-server                     Started         7.7s
    ❯ make stop  # Stop the stack. Data is preserved
    ❯ make down  # Delete the stack (containers, data, and networks)
    ```

Both options will expose the same server at <http://localhost:5000>. To set the tracking uri to the server:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
```

## Full Example

When developing a model, the training script should look something like this

```python
import mlflow
from sklearn import datasets, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from config import BaseConfig
from models.mlflow_wrappers import MLFlowConfig, mlflow_log_config


class Config(BaseConfig):
    mlflow: MLFlowConfig


config = Config(mlflow=MLFlowConfig(experiment_name="diabetes"))

diabetes = datasets.load_diabetes()

X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

mlflow.set_tracking_uri(config.mlflow.tracking_uri.get_secret_value())
mlflow.set_experiment(config.mlflow.experiment_name)

with mlflow.start_run(run_name=config.mlflow.run_name) as run:
    # NOTE: this will log mlflow metadata automatically in the run.
    # We disable it again to avoid logging the deep parameters of the pipeline
    mlflow.sklearn.autolog()
    mlflow.sklearn.autolog(disable=True)

    mlflow_log_config(config, artifact_path="config")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_metrics(
        {
            "r2_score": metrics.r2_score(y_pred, y_test),
            "mean_absolute_error": metrics.mean_absolute_error(y_pred, y_test),
        }
    )

    mlflow.sklearn.log_model(model, "model")
```

As you can see, we use the `BaseConfig` and the `MLFlowConfig` models to define the configuration and set the tracking URI properly. More about this in the [configuration](../../getting_started/01-configuration) section.
