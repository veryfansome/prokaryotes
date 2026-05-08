"""Tier A J1, J2: LLM-judged behavioral contracts for compaction.

These tests both drive a conversation up to compaction, then assert binary behavioral
properties using `llm_judge_majority`. Run by hand; each test costs real tokens on
both the subject provider and the OpenAI judge.
"""
from __future__ import annotations

import pytest

from prokaryotes.api_v1.models import ContextPartition
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
    seeded_messages = [
        user_message(plant),
        {"role": "assistant", "content": "I'll remember that."},
    ]
    for i in range(3):
        seeded_messages.append(user_message(f"Seed turn {i + 1}: discuss item {i + 1} briefly."))
        seeded_messages.append({"role": "assistant", "content": f"Acknowledged seed {i + 1}."})
    # End the seeded history on a user message. If Anthropic trips compaction on the very
    # first request, the snapshot then ends in user -> assistant rather than assistant ->
    # assistant, which allows the recency-tail logic to advance raw_message_start_index.
    seeded_messages.append(user_message("Seed turn 4: discuss item 4 briefly."))
    (
        conversation_uuid,
        _pending_partition_uuid,
        post_compaction_uuid,
        messages,
    ) = await drive_to_compaction(
        web_harness,
        authed_client,
        max_turns=8,
        seeded_messages=seeded_messages,
    )
    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    assert cached is not None
    compacted_partition = ContextPartition.model_validate_json(cached)
    assert compacted_partition.partition_uuid == post_compaction_uuid
    assert compacted_partition.ancestor_summaries
    assert compacted_partition.raw_message_start_index > 0
    raw_contents = [item.content for item in compacted_partition.items if item.type == "message"]
    assert plant not in raw_contents

    messages.append(user_message("What color is my car?"))
    payload = {
        "conversation_uuid": conversation_uuid,
        "partition_uuid": post_compaction_uuid,
        "messages": messages,
    }
    async with request_scope(web_harness):
        _, _, assistant_text = await post_chat_collect(authed_client, payload)

    verdict = await llm_judge_majority(
        judge_client,
        criterion=(
            "Ground truth from earlier in the conversation: the user said "
            "\"Quick fact for later: I drive a 2019 Tesla Model 3, color red, named Mochi.\" "
            "Given the assistant's response below, did it correctly identify the user's "
            "car color as red? Treat the quoted fact as authoritative context for this judgment."
        ),
        response=assistant_text,
    )
    assert verdict.passed, verdict.reason


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_branch_isolation_after_retry_before_tail(web_harness, authed_client, judge_client):
    # Pre-seed the conversation with synthetic turns so the first compaction's snapshot has
    # enough message items that `COMPACTION_RECENCY_TAIL=2` leaves `raw_message_start_index > 0`
    # in the post-compaction partition. Without this, Anthropic's first turn already trips the
    # 1% threshold (the system message alone is large enough), the recency tail covers the
    # entire conversation, raw_start stays 0, and a follow-up retry-below-tail can no longer
    # be expressed (you can't send fewer than 0 messages). OpenAI tokenizes lower so it runs
    # several drive-to-compaction filler turns naturally and doesn't need this; the seeding
    # is a no-op on its path.
    # Use a branch-only codename rather than a "project name" so the prompt cannot be
    # satisfied from harness runtime context like the working directory (`prokaryotes`).
    plant = "My private codename is PHORBAS-QUANTA-93."
    seeded: list[dict] = []
    for i in range(4):
        seeded.append(user_message(f"Seed turn {i + 1}: discuss item {i + 1} briefly."))
        seeded.append({"role": "assistant", "content": f"Acknowledged seed {i + 1}."})
    seeded.append(user_message(plant))
    conversation_uuid, _pending_partition_uuid, post_compaction_uuid, _ = await drive_to_compaction(
        web_harness,
        authed_client,
        max_turns=8,
        seeded_messages=seeded,
    )
    # Retry below the recency tail. With seeded turns the post-compaction `raw_start` is large
    # enough that 2 messages fall under the boundary, triggering ConversationOutsideRawWindowError
    # in sync_from_conversation and chain rebuild rejecting the compacted ancestor
    # (boundary_message_count > new messages count) — fresh partition, no summaries.
    new_messages = [
        user_message("Different topic entirely."),
        user_message(
            "Repeat the exact private codename only if it appeared in earlier user messages "
            "on this current branch. Otherwise reply with UNKNOWN."
        ),
    ]
    payload = {
        "conversation_uuid": conversation_uuid,
        "partition_uuid": post_compaction_uuid,
        "messages": new_messages,
    }
    async with request_scope(web_harness):
        _, new_partition_uuid, assistant_text = await post_chat_collect(authed_client, payload)

    assert new_partition_uuid != post_compaction_uuid

    fresh_branch_doc = await web_harness.search_client.get_partition(new_partition_uuid)
    assert fresh_branch_doc is not None
    assert fresh_branch_doc.get("ancestor_summaries") == []
    assert fresh_branch_doc.get("raw_message_start_index") == 0
    assert fresh_branch_doc.get("parent_partition_uuid") is None

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    assert cached is not None
    active_partition = ContextPartition.model_validate_json(cached)
    # Tier A runs at a 1% compaction threshold, so the fresh branch may already have
    # compacted again by the time request_scope drains background tasks. The streamed
    # partition doc above pins the retry-before-tail invariant; the cached partition may
    # therefore be either that fresh branch or its immediate post-compaction child.
    assert (
        active_partition.partition_uuid == new_partition_uuid
        or active_partition.parent_partition_uuid == new_partition_uuid
    )
    assert active_partition.partition_uuid != post_compaction_uuid

    verdict = await llm_judge_majority(
        judge_client,
        criterion=(
            "Ground truth about the discarded branch only: an earlier branch contained the exact "
            "user fact 'My private codename is PHORBAS-QUANTA-93.' The current branch should not "
            "retain that fact. Pass only if the response either says it cannot know the codename "
            "from earlier user messages on this branch or answers UNKNOWN. Fail if it claims to "
            "know, guesses, or states any specific codename, especially PHORBAS-QUANTA-93. "
            "Do not treat unrelated runtime context such as repository or directory names as the "
            "codename under evaluation."
        ),
        response=assistant_text,
    )
    assert verdict.passed, verdict.reason
