"""Root conftest for the integration tier.

Module-level env setup runs before any `prokaryotes` import. Both web harness modules pull
`COMPACTION_TOKEN_THRESHOLD_PCT` / `COMPACTION_RECENCY_TAIL` into their own namespace via `from
prokaryotes.utils_v1.llm_utils import …`, so the bindings freeze at import time. Setting env vars at module top —
above any `from prokaryotes…` import, with harness imports deferred into fixtures — is the only way to make the
override actually take effect.
"""

import os

import pytest
import pytest_asyncio

from tests.integration_tests.env_bootstrap import configure_integration_test_env

configure_integration_test_env()


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
