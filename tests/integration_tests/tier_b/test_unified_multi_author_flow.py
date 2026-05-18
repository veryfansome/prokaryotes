"""Tier B multi-author flow.

Two distinct human users posting into one `conversation_uuid`: the stored `Conversation` accumulates messages under
different `author_id`s, and `project_for_llm` prefixes each human turn with that author's display name.

The web surface is single-user per session, but the conversation model is multi-author by construction — this
exercises that path end to end through real syncer/storage, beyond the `project_for_llm` unit tests.
"""

from __future__ import annotations

import secrets
from uuid import uuid4

import httpx
import pytest
from httpx import ASGITransport

from prokaryotes.conversation_v1.models import Conversation
from prokaryotes.conversation_v1.project import project_for_llm
from tests.integration_tests.tier_b._helpers import post_chat_and_advance, user_message
from tests.unit_tests._llm_fakes import LLMRound, LLMScript

pytestmark = pytest.mark.integration


async def _register_client(web_harness, full_name: str) -> httpx.AsyncClient:
    """Register a fresh user and return an authed client.

    Mirrors the conftest `_authed_client_ctx` registration but takes a caller-chosen `full_name` so the two humans
    have distinct display names.
    """
    client = httpx.AsyncClient(transport=ASGITransport(app=web_harness.app), base_url="http://test")
    password = secrets.token_urlsafe(16)
    await client.post(
        "/register",
        data={
            "confirm_password": password,
            "email": f"{full_name.split()[0].lower()}-{uuid4()}@prokaryotes.test",
            "full_name": full_name,
            "password": password,
        },
    )
    return client


@pytest.mark.parametrize("web_harness", ["anthropic", "openai"], indirect=True)
@pytest.mark.asyncio(loop_scope="session")
async def test_two_humans_one_conversation_prefixes_display_names(web_harness):
    conversation_uuid = str(uuid4())
    alice = await _register_client(web_harness, "Alice Archaea")
    bob = await _register_client(web_harness, "Bob Bacterium")
    try:
        # Turn 1 — Alice opens the conversation.
        web_harness.llm_client.set_script(
            LLMScript(rounds=[LLMRound(text_deltas=["reply one"], stop_reason="end_turn")])
        )
        messages = [user_message("Alice here")]
        record1 = await post_chat_and_advance(web_harness, alice, conversation_uuid, messages)

        # Turn 2 — Bob continues the same conversation (same conversation_uuid + snapshot_uuid), echoing the history
        # Alice's client accumulated.
        web_harness.llm_client.set_script(
            LLMScript(rounds=[LLMRound(text_deltas=["reply two"], stop_reason="end_turn")])
        )
        messages.append(user_message("Bob here"))
        record2 = await post_chat_and_advance(
            web_harness, bob, conversation_uuid, messages, snapshot_uuid=record1.snapshot_uuid
        )
        assert record2.snapshot_uuid == record1.snapshot_uuid

        cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
        conv = Conversation.model_validate_json(cached)
        human = [m for m in conv.messages if m.author_id != conv.bot_author_id]
        # Reconcile preserved each message's original author — two distinct humans are stored even though every POST
        # re-stamps echoed entries with the posting session's author_id.
        assert len({m.author_id for m in human}) == 2

        # Projection derives multi-author mode and prefixes each human turn with that author's stored display name.
        projected = project_for_llm(conv)
        joined = "\n".join(p.content for p in projected if p.role == "user")
        assert "<Alice Archaea> Alice here" in joined
        assert "<Bob Bacterium> Bob here" in joined
    finally:
        await alice.aclose()
        await bob.aclose()
