FROM python:3.12.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

FROM base AS dependencies

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --prefix=/install .

FROM dependencies AS test-dependencies

RUN pip install --prefix=/test-install "pytest>=8.0"

FROM base AS production

COPY --from=dependencies /install /usr/local
COPY config ./config
RUN groupadd --system app && useradd --system --gid app --create-home app \
    && mkdir /input /output \
    && chown app:app /output

USER app

ENTRYPOINT ["umux-process"]

FROM production AS test

USER root
COPY --from=test-dependencies /test-install /usr/local
COPY tests ./tests
COPY compose.yaml Dockerfile ./
RUN mkdir /tmp/pytest-cache \
    && chown -R app:app /app/tests /tmp/pytest-cache
ENV PYTEST_ADDOPTS="-o cache_dir=/tmp/pytest-cache"
USER app

ENTRYPOINT ["pytest"]
CMD []
