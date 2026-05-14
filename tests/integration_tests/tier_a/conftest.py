"""Tier A fixtures: real-client harnesses gated on per-provider API keys."""
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
