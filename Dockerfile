FROM public.ecr.aws/docker/library/python:3.10-slim AS builder

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# hadolint ignore=DL3013
RUN pip install --no-cache-dir --upgrade pipenv==2022.1.8 pip

ENV PIPENV_VENV_IN_PROJECT=1
ENV VIRTUALENV_COPIES=1

WORKDIR /opt/project
COPY ./Pipfile.lock ./
COPY ./Pipfile ./
RUN pipenv sync

FROM public.ecr.aws/docker/library/python:3.10-slim AS mlflow

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/project
COPY --from=builder /opt/project/.venv .venv
ENV PATH="/opt/project/.venv/bin:$PATH"

ENTRYPOINT ["mlflow", "server"]

FROM public.ecr.aws/docker/library/python:3.10-slim AS runner

WORKDIR /opt/project
COPY --from=builder /opt/project/.venv .venv
ENV PATH="/opt/project/.venv/bin:$PATH"

COPY src ./

ENTRYPOINT ["python", "-m"]
