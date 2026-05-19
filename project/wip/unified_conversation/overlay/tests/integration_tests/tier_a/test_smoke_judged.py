"""Tier A J1, J2: LLM-judged behavioral contracts for compaction.

These tests drive a conversation up to compaction, then assert binary
behavioral properties using `llm_judge_majority`. Run by hand; each test
costs real tokens on both the subject provider and the OpenAI judge.
"""

from __future__ import annotations

import pytest

from prokaryotes.conversation_v1.models import Conversation
from tests.integration_tests.judges import llm_judge_majority
from tests.integration_tests.stream_utils import request_scope
from tests.integration_tests.tier_a._helpers import (
    drive_to_compaction,
    post_chat_collect,
    user_message,
)

pytestmark = [pytest.mark.live, pytest.mark.integration]


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_memory_continuity_across_compaction(web_harness, authed_client, judge_client):
    plant = "Quick fact for later: I drive a 2019 Tesla Model 3, color red, named Mochi."
    seeded_messages = [user_message(plant)]
    for i in range(3):
        seeded_messages.append(user_message(f"Seed turn {i + 1}: discuss item {i + 1} briefly."))
    seeded_messages.append(user_message("Seed turn 4: discuss item 4 briefly."))
    (
        conversation_uuid,
        _pending_snapshot_uuid,
        post_compaction_uuid,
        messages,
    ) = await drive_to_compaction(
        web_harness,
        authed_client,
        max_turns=8,
        seeded_messages=seeded_messages,
    )
    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    assert cached is not None
    compacted = Conversation.model_validate_json(cached)
    assert compacted.snapshot_uuid == post_compaction_uuid
    assert compacted.ancestor_summaries
    assert compacted.raw_message_start_index > 0
    raw_contents = [m.content for m in compacted.messages]
    assert plant not in raw_contents

    messages.append(user_message("What color is my car?"))
    payload = {
        "conversation_uuid": conversation_uuid,
        "snapshot_uuid": post_compaction_uuid,
        "messages": messages,
    }
    async with request_scope(web_harness):
        record = await post_chat_collect(authed_client, payload)

    verdict = await llm_judge_majority(
        judge_client,
        criterion=(
            "Ground truth from earlier in the conversation: the user said "
            '"Quick fact for later: I drive a 2019 Tesla Model 3, color red, named Mochi." '
            "Given the assistant's response below, did it correctly identify the user's "
            "car color as red? Treat the quoted fact as authoritative context for this judgment."
        ),
        response=record.assistant_text,
    )
    assert verdict.passed, verdict.reason


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_branch_isolation_after_retry_before_tail(web_harness, authed_client, judge_client):
    """Retry below the compacted recency tail → fresh branch, no carryover summaries."""
    plant = "My private codename is PHORBAS-QUANTA-93."
    seeded: list[dict] = []
    for i in range(4):
        seeded.append(user_message(f"Seed turn {i + 1}: discuss item {i + 1} briefly."))
    seeded.append(user_message(plant))
    conversation_uuid, _pending_snapshot_uuid, post_compaction_uuid, _ = await drive_to_compaction(
        web_harness,
        authed_client,
        max_turns=8,
        seeded_messages=seeded,
    )
    # Retry below the recency tail (only 2 new user messages, no echo of pre-compaction history).
    new_messages = [
        user_message("Different topic entirely."),
        user_message(
            "Repeat the exact private codename only if it appeared in earlier user messages "
            "on this current branch. Otherwise reply with UNKNOWN."
        ),
    ]
    payload = {
        "conversation_uuid": conversation_uuid,
        "snapshot_uuid": post_compaction_uuid,
        "messages": new_messages,
    }
    async with request_scope(web_harness):
        record = await post_chat_collect(authed_client, payload)

    assert record.snapshot_uuid != post_compaction_uuid

    fresh_branch_doc = await web_harness.search_client.get_conversation(record.snapshot_uuid)
    assert fresh_branch_doc is not None
    assert fresh_branch_doc.get("ancestor_summaries") == []
    assert fresh_branch_doc.get("raw_message_start_index") == 0
    assert fresh_branch_doc.get("parent_snapshot_uuid") is None

    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    assert cached is not None
    active = Conversation.model_validate_json(cached)
    # The fresh branch may immediately compact again at 1% threshold; the cached
    # snapshot may therefore be either it or its post-compaction child.
    assert active.snapshot_uuid == record.snapshot_uuid or active.parent_snapshot_uuid == record.snapshot_uuid
    assert active.snapshot_uuid != post_compaction_uuid

    verdict = await llm_judge_majority(
        judge_client,
        criterion=(
            "Ground truth about the discarded branch only: an earlier branch contained the exact "
            "user fact 'My private codename is PHORBAS-QUANTA-93.' The current branch should not "
            "retain that fact. Pass only if the response either says it cannot know the codename "
            "from earlier user messages on this branch or answers UNKNOWN. Fail if it claims to "
            "know, guesses, or states any specific codename, especially PHORBAS-QUANTA-93."
        ),
        response=record.assistant_text,
    )
    assert verdict.passed, verdict.reason
