"""Tier B scenario 8 (split): auth and validation boundaries on /chat."""
from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import pytest
from httpx import ASGITransport

pytestmark = pytest.mark.integration


@asynccontextmanager
async def _unauth_client(harness):
    transport = ASGITransport(app=harness.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.parametrize("web_harness", ["anthropic", "openai"], indirect=True)
@pytest.mark.asyncio(loop_scope="session")
async def test_chat_rejects_missing_session(web_harness):
    async with _unauth_client(web_harness) as client:
        response = await client.post(
            "/chat",
            json={"conversation_uuid": str(uuid4()), "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Session expired"


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_chat_rejects_empty_messages(web_harness, authed_client):
    response = await authed_client.post(
        "/chat",
        json={"conversation_uuid": str(uuid4()), "messages": []},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "At least one message is required"
