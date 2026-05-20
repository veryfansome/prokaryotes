"""Shared environment bootstrap for the integration test tiers."""
from __future__ import annotations

import os

from dotenv import load_dotenv


def configure_integration_test_env(*, running_in_docker: bool | None = None) -> None:
    """Populate datastore env vars for host-side and containerized integration runs.

    The checked-in `.env` uses Docker-network service names because the app normally runs inside compose. Host-side
    `uv run pytest tests/integration_tests/...` cannot resolve those names, so we intentionally rewrite the datastore
    endpoints to `localhost` there.

    Containerized integration runs are expected to receive the compose-style defaults from `.env` (or explicit env),
    so we leave those values alone in Docker.
    """
    load_dotenv()

    if running_in_docker is None:
        running_in_docker = os.path.exists("/.dockerenv")

    if not running_in_docker:
        os.environ["POSTGRES_HOST"] = "localhost"
        os.environ["POSTGRES_USER"] = "postgres"
        os.environ["POSTGRES_PASSWORD"] = "Ma9icMicr0be"
        os.environ["POSTGRES_DB"] = "prokaryotes"
        os.environ["REDIS_HOST"] = "localhost"
        os.environ["ELASTIC_URI"] = "http://localhost:9200"

    # Compaction tuning: a single global value satisfies both tiers — Tier B's fake controls scripted input_tokens;
    # Tier A needs `1` to trip in 2–3 normal turns. A per-tier split would silently break a combined run because the
    # harness modules cache the constants in their own namespace at import time.
    os.environ["COMPACTION_TOKEN_THRESHOLD_PCT"] = "1"
    os.environ["COMPACTION_RECENCY_TAIL"] = "2"
