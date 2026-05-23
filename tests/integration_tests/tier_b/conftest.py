"""Tier B fixtures: fake-backed harnesses, authed clients, autouse fake reset.

Uses the unit-tier LLM fakes (tests.unit_tests._llm_fakes) so the suite keeps exactly one fake-LLM contract.
"""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _web_harness_anthropic():
    from prokaryotes.harness_v1.web import WebHarness
    from tests.unit_tests._llm_fakes import FakeAnthropicClient

    harness = WebHarness(impl="anthropic", static_dir="scripts/static")
    harness.llm_client = FakeAnthropicClient()  # replace BEFORE init()
    harness.init()
    async with LifespanManager(harness.app):
        yield harness


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _web_harness_openai():
    from prokaryotes.harness_v1.web import WebHarness
    from tests.unit_tests._llm_fakes import FakeOpenAIClient

    harness = WebHarness(impl="openai", static_dir="scripts/static")
    harness.llm_client = FakeOpenAIClient()
    harness.init()
    async with LifespanManager(harness.app):
        yield harness


@pytest.fixture
def web_harness(request):
    return request.getfixturevalue(f"_web_harness_{request.param}")


@asynccontextmanager
async def _authed_client_ctx(harness):
    transport = ASGITransport(app=harness.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        password = secrets.token_urlsafe(16)
        await client.post(
            "/register",
            data={
                "confirm_password": password,
                "email": f"peter-{uuid4()}@prokaryotes.test",
                "full_name": "Peter Prokaryote",
                "password": password,
            },
        )
        yield client


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _authed_client_anthropic(_web_harness_anthropic):
    async with _authed_client_ctx(_web_harness_anthropic) as client:
        yield client


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _authed_client_openai(_web_harness_openai):
    async with _authed_client_ctx(_web_harness_openai) as client:
        yield client


@pytest.fixture
def authed_client(request):
    return request.getfixturevalue(f"_authed_client_{request.param}")


@pytest.fixture(autouse=True)
def _reset_fake_llm(request):
    """Reset fake LLM state before each test."""
    if "web_harness" not in request.fixturenames:
        yield
        return
    harness = request.getfixturevalue("web_harness")
    harness.llm_client.reset()
    yield


# -----------------------------------------------------------------------------
# Slack tier B fixtures
# -----------------------------------------------------------------------------
#
# The Slack tier B suite shares the same docker-compose stores as the web tier B suite. To avoid wiping web-
# harness state, cleanup is scoped to ES docs tagged with `bot_author_id="U_BOT"` (the Slack test bot identity;
# the web harness uses `__bot__`) and to the per-conversation Redis keys those docs imply. Non-Slack keys and
# docs are left alone.
#
# `_clear_slack_state_once_per_session` is *not* autouse — it runs once per session, pulled in transitively via
# `slack_harness`, so a pytest run that doesn't select any Slack tests pays no cost.


_SLACK_TEST_BOT_AUTHOR_ID = "U_BOT"


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _clear_slack_state_once_per_session():
    """Wipe only the Slack tier B test state — ES docs with `bot_author_id="U_BOT"` plus the per-conversation
    Redis keys those docs imply. Runs once at the start of any session that selects a Slack tier B test.
    """
    import os

    from elasticsearch import AsyncElasticsearch, NotFoundError
    from redis.asyncio import Redis

    redis = Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        db=int(os.environ.get("REDIS_DB", "0")),
    )
    es = AsyncElasticsearch(os.environ.get("ELASTIC_URI", "http://localhost:9200"))
    try:
        test_uuids: list[str] = []
        try:
            search = await es.search(
                index="conversations",
                query={"term": {"bot_author_id": _SLACK_TEST_BOT_AUTHOR_ID}},
                source=["conversation_uuid"],
                size=10_000,
            )
            test_uuids = list({hit["_source"]["conversation_uuid"] for hit in search["hits"]["hits"]})
        except NotFoundError:
            pass

        if test_uuids:
            redis_keys: list[str] = []
            for uuid_str in test_uuids:
                redis_keys.extend(
                    [
                        f"conversation:{uuid_str}",
                        f"assistant_index:{uuid_str}",
                        f"slack_prelude:{uuid_str}",
                        f"compaction_lock:{uuid_str}",
                    ]
                )
            if redis_keys:
                await redis.delete(*redis_keys)

        try:
            await es.delete_by_query(
                index="conversations",
                body={"query": {"term": {"bot_author_id": _SLACK_TEST_BOT_AUTHOR_ID}}},
                refresh=True,
                conflicts="proceed",
            )
        except NotFoundError:
            pass
        if test_uuids:
            try:
                await es.delete_by_query(
                    index="turn-executions",
                    body={"query": {"terms": {"conversation_uuid": test_uuids}}},
                    refresh=True,
                    conflicts="proceed",
                )
            except NotFoundError:
                pass
    finally:
        await redis.aclose()
        await es.close()
    yield


@pytest_asyncio.fixture(loop_scope="session")
async def slack_harness(_clear_slack_state_once_per_session):
    """A started `SlackTierBHarness` on real Redis + Elasticsearch with fake Slack / LLM boundaries.

    Function-scoped so each test gets a clean `_turn_locks` map and a fresh fake-LLM script; cross-run state is
    cleared once per session by `_clear_slack_state_once_per_session`.
    """
    import prokaryotes.slack_v1 as slack_v1
    from tests.integration_tests.tier_b._slack_tier_b import SlackTierBHarness
    from tests.unit_tests._slack_fakes import FakeSocketModeClient

    real_build = slack_v1.build_socket_mode_client
    slack_v1.build_socket_mode_client = lambda app_token: FakeSocketModeClient()
    harness = SlackTierBHarness()
    try:
        await harness.on_start()
        yield harness
    finally:
        await harness.on_stop()
        slack_v1.build_socket_mode_client = real_build


@pytest.fixture
def thread_client():
    """A fresh per-turn Slack Web API fake. Tests stage `thread` / `history` / `display_names` on it, then
    hand it to `slack_harness.deliver(...)` so the harness reconciles against it."""
    from tests.unit_tests._slack_fakes import FakeSlackThreadClient

    return FakeSlackThreadClient()
