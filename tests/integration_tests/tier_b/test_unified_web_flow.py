"""Tier B: plain (non-compacted) edit and regenerate web flows.

`test_compaction_flow.py` covers divergence across the compaction boundary; this covers edit and regenerate on a
short conversation that never compacted. Both diverge into a new branch snapshot (web apply policy) and must leave
the parent snapshot intact in ES.

Payloads are hand-built to model what the UI's edit / regenerate compose step puts on the wire: an edited message
arrives as a fresh node with no `source_id`; a regenerate re-POSTs history up to the user turn, dropping the bot
reply.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from prokaryotes.conversation_v1.models import Conversation
from prokaryotes.search_v1.conversations import messages_from_doc
from tests.integration_tests.tier_b._helpers import (
    post_chat,
    post_chat_and_advance,
    user_message,
)
from tests.unit_tests._llm_fakes import LLMRound, LLMScript

pytestmark = pytest.mark.integration


def _single_round(text: str) -> LLMScript:
    return LLMScript(rounds=[LLMRound(text_deltas=[text], stop_reason="end_turn")])


async def _establish_two_turns(web_harness, authed_client):
    """Build a non-compacted [u1, a1, u2, a2] conversation.

    Returns `(conversation_uuid, snapshot_uuid, messages)` where `messages` is the client's wire view with
    server-assigned source_ids stamped on.
    """
    conversation_uuid = str(uuid4())
    web_harness.llm_client.set_script(_single_round("a1"))
    messages = [user_message("u1")]
    record_1 = await post_chat_and_advance(web_harness, authed_client, conversation_uuid, messages)

    web_harness.llm_client.set_script(_single_round("a2"))
    messages.append(user_message("u2"))
    record_2 = await post_chat_and_advance(
        web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=record_1.snapshot_uuid
    )
    # A short conversation never compacts — both turns land on one snapshot.
    assert record_2.snapshot_uuid == record_1.snapshot_uuid
    return conversation_uuid, record_2.snapshot_uuid, messages


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_edit_earlier_user_message_branches_and_preserves_parent(web_harness, authed_client):
    """Editing u2 — fresh node with no source_id in u2's place, trailing a2 dropped — diverges into a new branch
    snapshot; parent stays intact in ES."""
    conversation_uuid, snapshot_uuid, messages = await _establish_two_turns(web_harness, authed_client)

    # UI edit: a freshly-authored node (no source_id) replaces u2; a2 is dropped.
    edited = [dict(messages[0]), dict(messages[1]), user_message("u2-edited")]
    web_harness.llm_client.set_script(_single_round("a2-fresh"))
    record = await post_chat(web_harness, authed_client, conversation_uuid, edited, snapshot_uuid=snapshot_uuid)

    branched_uuid = record.snapshot_uuid
    assert branched_uuid != snapshot_uuid

    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    conv = Conversation.model_validate_json(cached)
    assert conv.snapshot_uuid == branched_uuid
    assert conv.parent_snapshot_uuid == snapshot_uuid
    assert conv.ancestor_summaries == []  # never compacted
    assert [m.content for m in conv.messages] == ["u1", "a1", "u2-edited", "a2-fresh"]

    # The parent snapshot is untouched in ES — original u2 / a2 still present.
    parent_doc = await web_harness.search_client.get_conversation(snapshot_uuid)
    assert parent_doc is not None
    assert parent_doc["is_compacted"] is False
    assert [m.content for m in messages_from_doc(parent_doc)] == ["u1", "a1", "u2", "a2"]


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_regenerate_last_reply_branches_and_preserves_parent(web_harness, authed_client):
    """Regenerating a2 — the UI re-POSTs [u1, a1, u2] without the bot reply — diverges into a new branch snapshot
    with a fresh reply. The parent snapshot keeps the original a2 in ES."""
    conversation_uuid, snapshot_uuid, messages = await _establish_two_turns(web_harness, authed_client)

    # UI regenerate: re-POST the conversation up to u2, dropping the bot reply.
    regenerated = [dict(m) for m in messages[:3]]
    web_harness.llm_client.set_script(_single_round("a2-regenerated"))
    record = await post_chat(
        web_harness, authed_client, conversation_uuid, regenerated, snapshot_uuid=snapshot_uuid
    )

    branched_uuid = record.snapshot_uuid
    assert branched_uuid != snapshot_uuid

    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    conv = Conversation.model_validate_json(cached)
    assert conv.snapshot_uuid == branched_uuid
    assert conv.parent_snapshot_uuid == snapshot_uuid
    assert [m.content for m in conv.messages] == ["u1", "a1", "u2", "a2-regenerated"]

    # The parent snapshot still carries the original a2.
    parent_doc = await web_harness.search_client.get_conversation(snapshot_uuid)
    assert parent_doc is not None
    assert [m.content for m in messages_from_doc(parent_doc)] == ["u1", "a1", "u2", "a2"]
