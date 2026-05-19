"""Tier B fixtures: fake-backed harnesses, authed clients, autouse fake reset.

Uses the unit-tier-shared LLM fakes (tests.unit_tests._llm_fakes) so we keep
exactly one fake-LLM contract across the test suite.
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
    """Reset fake LLM state before each Tier B test."""
    if "web_harness" not in request.fixturenames:
        yield
        return
    harness = request.getfixturevalue("web_harness")
    harness.llm_client.reset()
    yield
