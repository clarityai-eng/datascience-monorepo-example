.DEFAULT_GOAL := help

PROJECT_NAME = datascience

DOCKER_IMAGE ?= $(PROJECT_NAME)
DOCKER_TAG ?= latest

ENV_FILE ?= .env


.PHONY: help
help: ## show help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: clean
clean:  ## Remove build artifacts
	@rm -fr build/
	@rm -fr dist/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +
	@find . -name '.hypothesis' -exec rm -fr {} +
	@find . -name '.ipynb_checkpoints' -exec rm -fr {} +
	@find . -name 'mlruns' -exec rm -fr {} +
	@rm -fr .tox/
	@rm -f .coverage
	@rm -f .coverage.*
	@rm -fr htmlcov/
	@rm -fr .pytest_cache
	@rm -fr .cache
	@rm -fr data/
	@rm -fr site/
	@PIPENV_VERBOSITY=-1 pipenv clean
	@docker compose down -v --remove-orphans

############################ Setup and Instalation ############################

.tmp: # Create tmp directory.
	@mkdir -p tmp/data

.env:  # Create .env files.
	@touch ${ENV_FILE}
	@echo "PIPENV_VERBOSITY=-1" >> ${ENV_FILE}

.PHONY: .install
.install: .env .tmp  ## Create tmp directory and .env files.

.PHONY: .install-precommit-hooks
.install-precommit-hooks: .install   ## Configure pre-commit hooks locally.
	@PIPENV_VERBOSITY=-1 pipenv run pre-commit install --install-hooks

setup: .install  ## Setup local environment installing both develop and default packages exactly as specified in `Pipfile.lock`.
	@pipenv sync --dev
	$(MAKE) .install-precommit-hooks all

.PHONY: update
update: .install  ## Updates pipenv environment
	@. ./${ENV_FILE} && pipenv update --dev
	@pipenv clean
	@PIPENV_VERBOSITY=-1 pipenv run pre-commit clean
	@PIPENV_VERBOSITY=-1 pipenv run pre-commit install-hooks

.PHONY: kernel
kernel: ## Create pipenv jupyter kernel
	@pipenv sync --dev
	@pipenv run python -m ipykernel install --user --name=$(PROJECT_NAME)

################################# Development #################################

PHONY: build
build:  .install  ## Build local image with DOCKER_TAG.
	@docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@docker compose build

.PHONY: test
test: .install   ## Locally run test
	@PIPENV_VERBOSITY=-1 pipenv run pytest
	@PIPENV_VERBOSITY=-1 pipenv run coverage html
	@PIPENV_VERBOSITY=-1 pipenv run coverage xml
	@rm .coverage.*

site:  ## Produce documentation.
	@PIPENV_VERBOSITY=-1 pipenv run mkdocs build

.PHONY: mkdocs
mkdocs:  ## Hot reaload documentation packages in your browser.
	@PIPENV_VERBOSITY=-1 pipenv run mkdocs serve

.PHONY: lint
lint: .install   ## Analyze the project files.
	@PIPENV_VERBOSITY=-1 pipenv run flake8 ./
	@docker run --rm -i hadolint/hadolint < Dockerfile

.PHONY: format
format: .install  ## Locally format files and imports.
	@PIPENV_VERBOSITY=-1 pipenv run black ./
	@PIPENV_VERBOSITY=-1 pipenv run pycln --all ./
	@PIPENV_VERBOSITY=-1 pipenv run isort --profile black ./

.PHONY: refurb
refurb: .install  ## Run refurb package to check for code upgrade suggestions
	@PIPENV_VERBOSITY=-1 pipenv run refurb ./src/ tests/

.PHONY: mypy
mypy: .install   ## Locally run mypy.
	@echo "Running mypy for Python typing checking"
	@PIPENV_VERBOSITY=-1 pipenv run pre-commit run mypy --all-files
	@echo "Done"

.PHONY: pre-commit
pre-commit:  ## Execute the precomit hooks
	@PIPENV_VERBOSITY=-1 pipenv run pre-commit run --all-files

.PHONY: all
all: format lint pre-commit test build site  ## Execute all make targets

################################## Services ##################################

up:  ## Run mlflow stack
	@docker compose up -d

stop:  ## Stop mlflow stack
	@docker compose up -d

down:  ## Remove containers, networks, and data for mlflow stack
	@docker compose down -v --remove-orphans
