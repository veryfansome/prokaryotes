FROM astral/uv:python3.12-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential

COPY pyproject.toml uv.lock     ./
COPY scripts                    ./scripts
COPY prokaryotes                ./prokaryotes

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra test --frozen --no-install-project
