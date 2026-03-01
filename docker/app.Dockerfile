FROM astral/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock     ./
COPY scripts                    ./scripts
COPY prokaryotes                ./prokaryotes

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --frozen --no-install-project
