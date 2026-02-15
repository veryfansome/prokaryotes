FROM astral/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra test --frozen --no-install-project

COPY scripts                    ./scripts
COPY prokaryotes                ./prokaryotes

ENTRYPOINT ["uv", "run"]
