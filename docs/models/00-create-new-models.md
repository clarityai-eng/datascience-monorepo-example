# Create new model modules

We want to encourage independence and adaptability of the project for every model use case. This means that there are no hard requirements for the model structure or practices as long as the input and output contracts are respected

In this page we recommend some set of practices and conventions to follow when creating new models.

## Create A New Model Module

The first step is to create a new module in the `src/models/` directory. The module name should be the same as the model name.

Typical modules to have are:

- `features.py`: extract and build features
- `preprocessing.py` (optional): make necessary preprocessing steps, like remove outliers, generate training split, etc.
- `train.py`: train the model with a set of parameters
- `tune.py` (optional): hyperparameter tunning of the model.

The model construction should be done in a separated module when possible (for example `src/models/iris/pipeline.py`). So this can be reused in the `train.py` or `tune.py` scripts.

You can include a `__main__.py` file in the model module to run all the steps in a single script. However, modularization of each step is recommended both for development and for production, as it creates smaller and independent tasks which can be fine tune.

## Create A New Model Documentation Page

The model documentation page is an important step of the model development. Is easy that this documentation is scattered in different docs, confluence pages, etc. To avoid this is encouraged that all critical information about the model is included in the model documentation page. This includes:

- Model name description
- Link to relevant MLFlow experiments, runs, and models
- Modeling decisions: features added or discarded, why we take an specific outlier removal strategy, etc.
- Links to presentations, Engineering Design Documents, Product Design Documents, etc.

The documentation of this repository should be focus on onboarding new users and developers as fast as possible. So it is encouraged to keep all technical documentation in the same place, this project docs. The exception to this rule is when creating materials to share with external stakeholders, like presentations to PR&I, Product, or Solutions teams.

## Integrate with MLFlow

Make sure that your model is integrated with MLFlow for experiment tracking and model registry. This is not enforced, but this integration is crucial for experiment reproducibility and deploy production models. Check the guide [MLFlow Usage](../guides/00-mlflow-usage) for more information
