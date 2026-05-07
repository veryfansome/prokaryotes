"""Tier A structural smoke (Anthropic). Real LLM, run by hand before release."""
from __future__ import annotations

from uuid import uuid4

import pytest

from prokaryotes.api_v1.models import ContextPartition
from tests.integration_tests.stream_utils import request_scope
from tests.integration_tests.tier_a._helpers import (
    drive_to_compaction,
    post_chat_collect,
    user_message,
)

pytestmark = [pytest.mark.live, pytest.mark.integration]


def _event_types(events: list[dict]) -> list[str]:
    return [k for ev in events for k in ev.keys()]


@pytest.mark.parametrize("web_harness, authed_client", [("anthropic", "anthropic")], indirect=True)
@pytest.mark.asyncio(loop_scope="session")
async def test_live_single_turn(web_harness, authed_client):
    payload = {
        "conversation_uuid": str(uuid4()),
        "messages": [user_message("Reply with the single word 'pong'.")],
    }
    async with request_scope(web_harness):
        events, partition_uuid, assistant_text = await post_chat_collect(authed_client, payload)
    types = _event_types(events)
    assert types[0] == "partition_uuid"
    assert types.count("context_pct") == 1
    assert types.index("context_pct") < types.index("text_delta")
    assert assistant_text  # non-empty
    cached = await web_harness.redis_client.get(f"context_partition:{payload['conversation_uuid']}")
    partition = ContextPartition.model_validate_json(cached)
    # Compaction may fire opportunistically at THRESHOLD_PCT=1 if the system message alone
    # already exceeds ~1% of the model's context window; in that case the cached partition
    # is the post-compaction child whose parent is the streamed UUID.
    assert partition.partition_uuid == partition_uuid or partition.parent_partition_uuid == partition_uuid
    doc = await web_harness.search_client.get_partition(partition_uuid)
    assert doc is not None


@pytest.mark.parametrize("web_harness, authed_client", [("anthropic", "anthropic")], indirect=True)
@pytest.mark.asyncio(loop_scope="session")
async def test_live_forced_compaction(web_harness, authed_client):
    (
        conversation_uuid,
        pending_partition_uuid,
        post_compaction_uuid,
        messages,
    ) = await drive_to_compaction(
        web_harness,
        authed_client,
        "Quick fact for later: my favorite color is teal.",
        max_turns=8,
    )
    payload = {
        "conversation_uuid": conversation_uuid,
        "partition_uuid": post_compaction_uuid,
        "messages": messages + [user_message("Reply with one short sentence.")],
    }
    async with request_scope(web_harness):
        _, partition_uuid, _ = await post_chat_collect(authed_client, payload)

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    partition = ContextPartition.model_validate_json(cached)
    # Same opportunistic-compaction caveat as test_live_single_turn.
    assert partition.partition_uuid == partition_uuid or partition.parent_partition_uuid == partition_uuid
    assert partition.partition_uuid != pending_partition_uuid
    assert partition.ancestor_summaries


@pytest.mark.parametrize("web_harness, authed_client", [("anthropic", "anthropic")], indirect=True)
@pytest.mark.asyncio(loop_scope="session")
async def test_live_tool_call_best_effort(web_harness, authed_client):
    payload = {
        "conversation_uuid": str(uuid4()),
        "messages": [user_message("Use the shell_command tool to run `echo hi`.")],
    }
    async with request_scope(web_harness):
        events, _, _ = await post_chat_collect(authed_client, payload)
    types = _event_types(events)
    assert types[0] == "partition_uuid"
    if "tool_call" in types:
        for ev in events:
            if "tool_call" in ev:
                assert isinstance(ev["tool_call"]["name"], str)
                assert isinstance(ev["tool_call"]["arguments"], str)
