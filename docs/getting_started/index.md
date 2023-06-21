# Getting Started

## Requirements

- Python >= 3.10 (can be installed with pyenv or with asdf)
- [Pipenv](https://pipenv.pypa.io/en/latest/) == 2022.9.24
- [direnv](https://direnv.net/)
- [Docker](https://www.docker.com/)

## Setup

The project is automated using `make` and the [`Makefile`](Makefile)

In the first run, use `make setup` to bootstrap the project setup in your local. This will install dependencies using pipenv and create the necessary files. It will also run `make all` for you so you can verify that everything is working in your local.

By the end of the setup you should be able to run `make all` successfully.

## `pre-commit` hooks

The project implements a series of pre-commit hooks to ensure consistency in codestyle and prevent broken commits. This can prevent commits, but most of the hooks fix the files themselves. If the commit is failing for you, make sure to `git add .` again to add the new changes files.

The commit hooks will also run integration tests to make sure that the contract between the modules are still working.

> NOTE: ruby version installed with precommit can raise compiling errors depending on your openssl local installation. To overcome this set `OPENSSL_CFLAGS=-Wno-error=implicit-function-declaration` environment variable in your `.env` before the setup. Source [1](https://github.com/openssl/openssl/issues/18720#issuecomment-1180702773) and [2](https://dev.to/yasuhiron777/install-2x-3x-version-of-ruby-via-rbenv-on-m1-mac-3okn)

## Credentials and other environment variables

Environment variables will be picked up from the `.env` file. This file is ignored by git, so you can add your own variables there. The `.env` file is automatically loaded by `direnv` when you enter the project directory. If you experience issues with `direnv`, make sure that you have allowed the directory with `direnv allow`.

Typical environment variables you will need to setup:

- `SNOWFLAKE_URL`: access to snowflake for reading data.
- `MLFLOW_TRACKING_URL`: access to mlflow for logging metrics and models. You can set this url to a local mlflow server for testing (<http://localhost:5000>) or even point to a local folder like `./mlruns`. Also, as we are using Pydantic Settings objects for the config, you can override all the mlflow config using the format: `MLFLOW='{"tracking_uri": "http://localhost:5000"}'` in your `.env` file.

You can find more detailed information about the environment variables in the [configuration guide](01-configuration.md) module.

## Troubleshooting Common Problems

### pre-commit ruby errors

We use a pre-commit script to format and lint markdown that requires ruby. This causes the pre-commit installation during `make setup` to fail for some users on MacOS. If that is the case, try setting up the following env variable:

```env
OPENSSL_CFLAGS=-Wno-error=implicit-function-declaration
```

That should correct openssl errors on ruby instalations.
