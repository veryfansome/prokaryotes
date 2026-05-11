FROM astral/uv:python3.12-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    net-tools \
    ripgrep \
    telnet \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra test --frozen --no-install-project

COPY scripts                    ./scripts
COPY prokaryotes                ./prokaryotes

ENTRYPOINT ["uv", "run"]
