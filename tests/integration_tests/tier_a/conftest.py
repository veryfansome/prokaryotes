"""Tier A fixtures: real-client harnesses gated on per-provider API keys."""
from __future__ import annotations

import asyncio
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _web_harness_anthropic(live_keys_anthropic):
    from prokaryotes.harness_v1.web import WebHarness

    harness = WebHarness(impl="anthropic", static_dir="scripts/static")
    harness.init()
    async with LifespanManager(harness.app):
        yield harness


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _web_harness_openai(live_keys_openai):
    from prokaryotes.harness_v1.web import WebHarness

    harness = WebHarness(impl="openai", static_dir="scripts/static")
    harness.init()
    async with LifespanManager(harness.app):
        yield harness


@pytest.fixture
def web_harness(request):
    return request.getfixturevalue(f"_web_harness_{request.param}")


@asynccontextmanager
async def _authed_client_ctx(harness):
    transport = ASGITransport(app=harness.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120.0) as client:
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


# -----------------------------------------------------------------------------
# Slack tier A fixtures and helpers
# -----------------------------------------------------------------------------
#
# These exercise the live Slack API and a live LLM provider. Skipped unless the four Slack env vars
# (`SLACK_APP_TOKEN`, `SLACK_BOT_TOKEN`, `SLACK_USER_TOKEN`, `SLACK_TEST_CHANNEL`) and `ANTHROPIC_API_KEY` are
# all set. `live_keys_anthropic` is provided by the parent `tests/integration_tests/conftest.py`; the
# compaction-threshold tuning is set globally by `env_bootstrap.configure_integration_test_env` and is left
# alone here (Tier A's stated intent is to trip compaction in 2–3 turns).


@pytest.fixture(scope="session")
def slack_live_env():
    """Skip the Slack Tier A suite unless all four Slack env vars are set, then return them as a dict."""
    env_to_key = {
        "SLACK_APP_TOKEN": "app_token",
        "SLACK_BOT_TOKEN": "bot_token",
        "SLACK_USER_TOKEN": "user_token",
        "SLACK_TEST_CHANNEL": "channel",
    }
    values: dict[str, str | None] = {key: os.environ.get(env) for env, key in env_to_key.items()}
    missing = [env for env, key in env_to_key.items() if not values[key]]
    if missing:
        pytest.skip(f"Slack Tier A requires env vars: {', '.join(missing)}")
    return values


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def live_slack_harness(slack_live_env, request):
    """A real `SlackHarness` connected via Socket Mode. `SLACK_HARNESS_IMPL` (`anthropic` | `openai`, default
    `anthropic`) selects the LLM client, mirroring `scripts/slack.py`'s runtime selection; the matching
    `live_keys_<impl>` fixture is resolved via `request.getfixturevalue` so its skip-on-missing-key guard runs
    only for the provider actually exercised. Session-scoped so the Socket Mode handshake is paid for once across
    the suite."""
    from prokaryotes.harness_v1.slack import SlackHarness

    impl = os.environ.get("SLACK_HARNESS_IMPL", "anthropic")
    if impl not in {"anthropic", "openai"}:
        pytest.skip(f"unsupported SLACK_HARNESS_IMPL={impl!r}")
    # pytest can't dynamically declare fixture deps; getfixturevalue resolves the provider's live_keys gate at
    # runtime so a missing key skips cleanly instead of being reported as a collection error.
    request.getfixturevalue(f"live_keys_{impl}")

    harness = SlackHarness(
        impl=impl,
        app_token=slack_live_env["app_token"],
        bot_token=slack_live_env["bot_token"],
    )
    await harness.on_start()
    try:
        yield harness
    finally:
        await harness.on_stop()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def slack_test_user_id(slack_live_env):
    """Resolve the Slack user ID behind `SLACK_USER_TOKEN` via `auth.test`. The bot's reply prefix is
    `<@slack_test_user_id>`; tests assert on it."""
    body = await slack_api_call("auth.test", token=slack_live_env["user_token"], data={})
    assert body.get("ok"), f"auth.test on SLACK_USER_TOKEN failed: {body.get('error')}"
    return body["user_id"]


@pytest_asyncio.fixture
async def slack_messages_to_cleanup():
    """Best-effort per-test cleanup of messages the test posted. Tests append `(channel, ts, token)` tuples;
    teardown issues `chat.delete` for each, swallowing failures (rate-limit, already-gone, etc.)."""
    posted: list[tuple[str, str, str]] = []
    yield posted
    async with httpx.AsyncClient(base_url="https://slack.com/api/", timeout=20.0) as http:
        for channel, ts, token in posted:
            try:
                await http.post(
                    "chat.delete",
                    data={"channel": channel, "ts": ts, "token": token},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
            except Exception:
                pass


async def slack_api_call(method: str, *, token: str, data: dict[str, Any]) -> dict:
    """Raw form-encoded POST to Slack's Web API, mirroring `SlackClient._call`'s wire shape. Used by tests to
    post as the test user and to poll for bot replies, without re-curring through the harness's limiter."""
    async with httpx.AsyncClient(base_url="https://slack.com/api/", timeout=20.0) as http:
        response = await http.post(
            method,
            data={**data, "token": token},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        return response.json()


async def wait_for_bot_reply(
    *,
    after_ts: str = "0",
    bot_token: str,
    bot_user_id: str,
    channel: str,
    parent_ts: str,
    poll_interval: float = 2.0,
    timeout: float = 90.0,
) -> dict:
    """Poll `conversations.replies` until a finalized bot reply appears under `parent_ts`. Filters out the
    placeholder rewrite (`_…working_`) and anything at or before `after_ts`. Raises `TimeoutError` if no
    finalized reply lands within `timeout` seconds.

    Uses the bot token rather than the user token: reading thread replies requires `channels:history` (or its
    group/mpim/im siblings), which is on the bot token by design — `SLACK_USER_TOKEN` only needs `chat:write`
    for the "post as the test user" half of the test, so a misconfigured user token would otherwise silently
    `ok:false` every poll and look like a hang.
    """
    from prokaryotes.slack_v1.streaming import SlackStreamer

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = await slack_api_call(
            "conversations.replies",
            token=bot_token,
            data={"channel": channel, "ts": parent_ts},
        )
        if body.get("ok"):
            for m in body.get("messages", []):
                if m.get("ts") == parent_ts:
                    continue
                if m.get("ts", "") <= after_ts:
                    continue
                if m.get("user") != bot_user_id:
                    continue
                text = m.get("text") or ""
                if SlackStreamer.PLACEHOLDER_TEXT in text:
                    continue
                return m
        await asyncio.sleep(poll_interval)
    raise TimeoutError(f"No finalized bot reply for parent_ts={parent_ts} within {timeout}s")
