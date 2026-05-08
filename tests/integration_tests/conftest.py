"""Root conftest for the integration tier.

Module-level env setup runs before any `prokaryotes` import. Both web harness modules
pull `COMPACTION_TOKEN_THRESHOLD_PCT` / `COMPACTION_RECENCY_TAIL` into their own namespace
via `from prokaryotes.utils_v1.llm_utils import …`, so the bindings freeze at import time.
Setting env vars at module top — above any `from prokaryotes…` import, with harness imports
deferred into fixtures — is the only way to make the override actually take effect.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# `.env` and `.env.example` use Docker-network service names because the prod app runs
# inside the compose network. Host-side `uv run pytest` cannot resolve those names, so we
# rewrite them to localhost-mapped ports. In CI there may be no `.env` at all, so we also
# provide the compose-default Postgres credentials that the host-side test process needs.
# Direct assignment overrides any value already loaded from `.env`.
os.environ["POSTGRES_HOST"] = "localhost"
os.environ["POSTGRES_USER"] = "postgres"
os.environ["POSTGRES_PASSWORD"] = "Ma9icMicr0be"
os.environ["POSTGRES_DB"] = "prokaryotes"
os.environ["REDIS_HOST"] = "localhost"
os.environ["ELASTIC_URI"] = "http://localhost:9200"

# Compaction tuning: a single global value satisfies both tiers — Tier B's fake controls
# scripted input_tokens; Tier A needs `1` to trip in 2–3 normal turns. A per-tier split
# would silently break a combined run because the harness modules cache the constants
# in their own namespace at import time.
os.environ["COMPACTION_TOKEN_THRESHOLD_PCT"] = "1"
os.environ["COMPACTION_RECENCY_TAIL"] = "2"

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402


@pytest.fixture(scope="session")
def live_keys_anthropic():
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


@pytest.fixture(scope="session")
def live_keys_openai():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def judge_client(live_keys_openai):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        yield client
    finally:
        await client.close()
