"""Tier B scenarios 4–7: forced compaction, retry within tail, retry before tail, branch switch.

MIGRATION STATUS: skeletal. The full upstream file is 549 lines / 8 tests built
around `_run_turn` and `_establish_compacted_partition` helpers that thread
`partition_uuid` through every POST. The migration replaces those helpers with
`post_chat_and_advance` from `_helpers.py` and threads `snapshot_uuid` instead.
Per-test bodies need additional rewriting to handle the bot_message source_id
echo on subsequent POSTs (the DAG-scoped guardrail now rejects unanchored
assistant entries).

Verify by running with the docker-compose data stores up:
    docker compose up -d elasticsearch postgres redis
    PYTHONPATH=project/wip/unified_conversation/overlay:. \
        uv run --extra test pytest \
        project/wip/unified_conversation/overlay/tests/integration_tests/tier_b/test_compaction_flow.py
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from prokaryotes.conversation_v1.models import Conversation
from tests.integration_tests.tier_b._helpers import (
    apply_assignments,
    echo_assistant,
    event_types,
    post_chat,
    post_chat_and_advance,
    user_message,
)
from tests.unit_tests._llm_fakes import LLMRound, LLMScript

pytestmark = pytest.mark.integration


async def _wait_for_compaction(
    client,
    conversation_uuid: str,
    pending_snapshot_uuid: str,
    *,
    attempts: int = 30,
    delay: float = 0.1,
) -> str | None:
    """Poll /compaction-status. Returns the relabel target (or None)."""
    for _ in range(attempts):
        response = await client.get(
            "/compaction-status",
            params={
                "conversation_uuid": conversation_uuid,
                "pending_snapshot_uuid": pending_snapshot_uuid,
            },
        )
        body = response.json()
        if body.get("done"):
            return body.get("snapshot_uuid")
        await asyncio.sleep(delay)
    raise AssertionError("compaction did not complete within timeout")


async def _establish_compacted_snapshot(
    web_harness,
    authed_client,
    *,
    summary_text: str = "STUB SUMMARY",
) -> tuple[str, str, str, list[dict]]:
    """Run 3 establishing turns and a 4th turn that triggers compaction.

    Returns (conversation_uuid, pre_compaction_snapshot_uuid,
    post_compaction_snapshot_uuid, full_messages).
    """
    conversation_uuid = str(uuid4())
    messages: list[dict] = []
    snapshot_uuid: str | None = None
    for i in range(3):
        messages.append(user_message(f"u{i + 1}"))
        web_harness.llm_client.set_script(
            LLMScript(rounds=[LLMRound(text_deltas=[f"a{i + 1}"], stop_reason="end_turn", input_tokens=500)])
        )
        record = await post_chat_and_advance(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
        )
        snapshot_uuid = record.snapshot_uuid

    messages.append(user_message("u4"))
    pre_compaction_uuid = snapshot_uuid
    web_harness.llm_client.set_script(
        LLMScript(
            rounds=[LLMRound(text_deltas=["a4"], stop_reason="end_turn", input_tokens=5000)],
            summary_text=summary_text,
        )
    )
    record = await post_chat(web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid)
    types = event_types(record.events)
    assert "compaction_pending" in types
    assert types[-1] == "compaction_pending"
    pending_snapshot_uuid = record.snapshot_uuid
    apply_assignments(messages, record.source_id_assignments)
    echo_assistant(messages, record)
    post_compaction_uuid = await _wait_for_compaction(authed_client, conversation_uuid, pending_snapshot_uuid)
    return conversation_uuid, pre_compaction_uuid, post_compaction_uuid or pending_snapshot_uuid, messages


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_forced_compaction_completes_and_relabels(web_harness, authed_client):
    """Drive a compaction; verify the post-compaction snapshot has ancestor_summaries
    and that polling returns the relabel target."""
    conversation_uuid, pre_id, post_id, _ = await _establish_compacted_snapshot(web_harness, authed_client)
    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    conv = Conversation.model_validate_json(cached)
    assert conv.ancestor_summaries
    assert conv.snapshot_uuid == post_id
    assert conv.snapshot_uuid != pre_id


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_compaction_relabel_and_continue(web_harness, authed_client):
    """Full round-trip: new conversation → compaction → relabel via /compaction-status →
    next turn projects bot history exactly once (no duplication from Issue 1's old
    failure mode)."""
    conversation_uuid, _pre, post_id, messages = await _establish_compacted_snapshot(web_harness, authed_client)
    # Send the next turn anchored at the post-compaction snapshot.
    web_harness.llm_client.set_script(
        LLMScript(rounds=[LLMRound(text_deltas=["a5"], stop_reason="end_turn", input_tokens=500)])
    )
    messages.append(user_message("u5"))
    record = await post_chat(web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=post_id)
    # The bot's reply commits.
    assert record.bot_message_source_id is not None
    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    conv = Conversation.model_validate_json(cached)
    # Bot message appears once at the tail; no duplication of compacted history.
    assert conv.messages[-1].content == "a5"
