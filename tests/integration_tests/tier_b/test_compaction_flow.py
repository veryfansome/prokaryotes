"""Tier B compaction flow tests.

Forced compaction + relabel + continuation, post-summary-delay message carry-forward, multi-generation summary
accumulation, rebuild-from-compacted-ancestor after Redis miss (single + multi-gen), divergence within/before the
raw tail, and branch switch.

All tests anchor on `snapshot_uuid` and use `post_chat_and_advance` to keep client state aligned with the server's
source_id assignments (the DAG-scoped guardrail rejects assistant entries without a known source_id).
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from prokaryotes.conversation_v1.models import Conversation
from tests.integration_tests.stream_utils import collect_stream
from tests.integration_tests.tier_b._helpers import (
    TurnRecord,
    apply_assignments,
    echo_assistant,
    event_types,
    is_handshake,
    post_chat,
    post_chat_and_advance,
    user_message,
)
from tests.unit_tests._llm_fakes import LLMRound, LLMScript

pytestmark = pytest.mark.integration


async def _post_chat_without_request_scope(
    authed_client,
    conversation_uuid: str,
    messages: list[dict],
    *,
    snapshot_uuid: str | None = None,
) -> TurnRecord:
    """Like `post_chat` but does NOT wrap in `request_scope`. The caller owns the background compaction task this
    turn spawns — use when a test must race a follow-up turn against an in-flight summary."""
    payload: dict = {"conversation_uuid": conversation_uuid, "messages": messages}
    if snapshot_uuid is not None:
        payload["snapshot_uuid"] = snapshot_uuid
    async with authed_client.stream("POST", "/chat", json=payload) as response:
        assert response.status_code == 200
        events = await collect_stream(response)
    assert is_handshake(events[0]), f"first event not a handshake: {events[0]}"
    record = TurnRecord(
        snapshot_uuid=events[0]["snapshot_uuid"],
        source_id_assignments=events[0].get("source_id_assignments", []),
        events=events,
    )
    record.assistant_text = "".join(e["text_delta"] for e in events if "text_delta" in e)
    for ev in events:
        if "bot_message" in ev:
            record.bot_message_source_id = ev["bot_message"]["source_id"]
    return record


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
    summary_delay: float = 0.0,
) -> tuple[str, str, str, list[dict]]:
    """Run 3 establishing turns and a 4th turn that triggers compaction.

    Returns (conversation_uuid, pre_compaction_snapshot_uuid, post_compaction_snapshot_uuid, full_messages).
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
            summary_delay=summary_delay,
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


async def _establish_two_generation_compacted_snapshot(
    web_harness,
    authed_client,
) -> tuple[str, str, str, list[dict]]:
    """Two consecutive compactions. Returns (conversation_uuid,
    second_pending_snapshot_uuid, active_snapshot_uuid_after_gen2, messages)."""
    conversation_uuid, _pre, active_uuid, messages = await _establish_compacted_snapshot(
        web_harness, authed_client, summary_text="GEN-1"
    )

    # u5 lands on the post-GEN-1 snapshot in place (normal append).
    messages.append(user_message("u5"))
    web_harness.llm_client.set_script(
        LLMScript(rounds=[LLMRound(text_deltas=["a5"], stop_reason="end_turn", input_tokens=500)])
    )
    record = await post_chat_and_advance(
        web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=active_uuid
    )
    assert record.snapshot_uuid == active_uuid

    # u6 triggers the second compaction.
    messages.append(user_message("u6"))
    web_harness.llm_client.set_script(
        LLMScript(
            rounds=[LLMRound(text_deltas=["a6"], stop_reason="end_turn", input_tokens=5000)],
            summary_text="GEN-2",
        )
    )
    record = await post_chat(web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=active_uuid)
    types = event_types(record.events)
    assert "compaction_pending" in types
    assert types[-1] == "compaction_pending"
    second_pending_uuid = record.snapshot_uuid
    apply_assignments(messages, record.source_id_assignments)
    echo_assistant(messages, record)
    post_second_compaction_uuid = await _wait_for_compaction(authed_client, conversation_uuid, second_pending_uuid)

    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    conv = Conversation.model_validate_json(cached)
    assert conv.snapshot_uuid == post_second_compaction_uuid
    assert conv.parent_snapshot_uuid == second_pending_uuid
    assert conv.ancestor_summaries == ["GEN-1", "GEN-2"]
    return conversation_uuid, second_pending_uuid, post_second_compaction_uuid, messages


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_forced_compaction_completes_and_relabels(web_harness, authed_client):
    """Post-compaction snapshot has ancestor_summaries; polling returns the relabel target."""
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
    """Round-trip: compaction → relabel via /compaction-status → next turn projects bot history exactly once (Issue
    1's old failure mode duplicated it)."""
    conversation_uuid, _pre, post_id, messages = await _establish_compacted_snapshot(web_harness, authed_client)
    web_harness.llm_client.set_script(
        LLMScript(rounds=[LLMRound(text_deltas=["a5"], stop_reason="end_turn", input_tokens=500)])
    )
    messages.append(user_message("u5"))
    record = await post_chat(web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=post_id)
    assert record.bot_message_source_id is not None
    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    conv = Conversation.model_validate_json(cached)
    assert conv.messages[-1].content == "a5"


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_forced_compaction_marks_parent_and_redirects_stale_anchor(web_harness, authed_client):
    """Beyond the relabel: parent doc is marked is_compacted with summary + boundary fields, and a follow-up turn
    anchored at the (stale) pre-compaction snapshot still lands on the post-compaction snapshot via the relabel
    path."""
    conversation_uuid, pre_id, post_id, messages = await _establish_compacted_snapshot(web_harness, authed_client)

    parent_doc = await web_harness.search_client.get_conversation(pre_id)
    assert parent_doc is not None
    assert parent_doc["is_compacted"] is True
    assert parent_doc["summary"]
    assert parent_doc["boundary_hash"]
    assert parent_doc["tail_hash"]
    assert parent_doc["boundary_message_count"] > 0

    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    conv = Conversation.model_validate_json(cached)
    assert conv.snapshot_uuid == post_id
    assert conv.parent_snapshot_uuid == pre_id
    assert conv.ancestor_summaries == ["STUB SUMMARY"]
    assert conv.raw_message_start_index > 0

    # Follow-up anchored at the stale pre-compaction snapshot. The syncer must detect the relabel and land at
    # post-compaction.
    messages.append(user_message("after-compaction"))
    web_harness.llm_client.set_script(
        LLMScript(rounds=[LLMRound(text_deltas=["ok"], stop_reason="end_turn", input_tokens=500)])
    )
    record = await post_chat(web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=pre_id)
    assert record.snapshot_uuid == post_id


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_compaction_carries_forward_messages_added_during_summary(web_harness, authed_client):
    """A turn that arrives while the summary is still in flight commits to the
    pending snapshot in place; the eventual compaction must carry those post-snapshot messages into the child."""
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

    # One script up front covering both turns: set_script is monotonic state on the fake LLM, so calling it again
    # between u4 and u5 would race the bg compaction's `complete()` (reads summary_text after sleep).
    # summary_delay=2.0 keeps the summary call sleeping past the follow-up POST.
    messages.append(user_message("u4"))
    web_harness.llm_client.set_script(
        LLMScript(
            rounds=[
                LLMRound(text_deltas=["a4"], stop_reason="end_turn", input_tokens=5000),
                LLMRound(text_deltas=["a5"], stop_reason="end_turn", input_tokens=500),
            ],
            summary_delay=2.0,
            summary_text="GEN-1",
        )
    )
    # Bypass `request_scope` so the background compaction task isn't awaited before the follow-up POST runs.
    record = await _post_chat_without_request_scope(
        authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
    )
    types = event_types(record.events)
    assert "compaction_pending" in types
    assert types[-1] == "compaction_pending"
    pending_snapshot_uuid = record.snapshot_uuid
    apply_assignments(messages, record.source_id_assignments)
    echo_assistant(messages, record)

    # u5 anchored at the pending snapshot — should append in place because the compaction summary is still in flight
    # and the CAS swap hasn't run yet.
    messages.append(user_message("u5"))
    follow_up = await post_chat_and_advance(
        web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=pending_snapshot_uuid
    )
    assert follow_up.snapshot_uuid == pending_snapshot_uuid
    assert "compaction_pending" not in event_types(follow_up.events)

    post_compaction_uuid = await _wait_for_compaction(authed_client, conversation_uuid, pending_snapshot_uuid)
    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    conv = Conversation.model_validate_json(cached)
    assert conv.snapshot_uuid == post_compaction_uuid
    assert conv.parent_snapshot_uuid == pending_snapshot_uuid
    assert conv.ancestor_summaries == ["GEN-1"]
    contents = [m.content for m in conv.messages]
    assert contents[-4:] == ["u4", "a4", "u5", "a5"]


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_compaction_accumulates_ancestor_summaries_across_generations(web_harness, authed_client):
    """Two compactions in sequence — the child of gen-2 carries both summaries in order."""
    (
        conversation_uuid,
        second_pending_uuid,
        active_uuid,
        _messages,
    ) = await _establish_two_generation_compacted_snapshot(web_harness, authed_client)

    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    conv = Conversation.model_validate_json(cached)
    assert conv.snapshot_uuid == active_uuid
    assert conv.snapshot_uuid != second_pending_uuid
    assert conv.parent_snapshot_uuid == second_pending_uuid
    assert conv.ancestor_summaries == ["GEN-1", "GEN-2"]


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_rebuild_from_compacted_ancestor_after_redis_miss(web_harness, authed_client):
    """After compaction, evict Redis and post anchored at the stale pre-compaction snapshot. The syncer must walk
    the chain via ES, find the compacted ancestor, and rebuild a fresh active snapshot that inherits the ancestor's
    summaries."""
    conversation_uuid, pre_id, post_id, messages = await _establish_compacted_snapshot(web_harness, authed_client)

    await web_harness.redis_client.delete(f"conversation:{conversation_uuid}")

    messages.append(user_message("after-redis-miss"))
    web_harness.llm_client.set_script(
        LLMScript(rounds=[LLMRound(text_deltas=["restored"], stop_reason="end_turn", input_tokens=500)])
    )
    record = await post_chat(web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=pre_id)
    rebuilt_uuid = record.snapshot_uuid
    assert rebuilt_uuid != pre_id
    assert rebuilt_uuid != post_id

    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    assert cached is not None
    conv = Conversation.model_validate_json(cached)
    assert conv.snapshot_uuid == rebuilt_uuid
    assert conv.ancestor_summaries == ["STUB SUMMARY"]
    assert conv.raw_message_start_index > 0
    # `conv.messages` holds only the raw-window tail (everything after `raw_message_start_index` of the full
    # reconstructed history).
    tail_contents = [m.content for m in conv.messages]
    assert "after-redis-miss" in tail_contents
    assert "restored" in tail_contents

    doc = await web_harness.search_client.get_conversation(rebuilt_uuid)
    assert doc is not None
    assert doc["is_compacted"] is False


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_rebuild_from_multigeneration_compacted_ancestor_after_redis_miss(web_harness, authed_client):
    """Same rebuild semantics across two compaction generations: the rebuilt snapshot inherits both summaries."""
    (
        conversation_uuid,
        second_pending_uuid,
        post_second_compaction_uuid,
        messages,
    ) = await _establish_two_generation_compacted_snapshot(web_harness, authed_client)

    await web_harness.redis_client.delete(f"conversation:{conversation_uuid}")

    messages.append(user_message("after-second-redis-miss"))
    web_harness.llm_client.set_script(
        LLMScript(rounds=[LLMRound(text_deltas=["restored-again"], stop_reason="end_turn", input_tokens=500)])
    )
    record = await post_chat(web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=second_pending_uuid)
    rebuilt_uuid = record.snapshot_uuid
    assert rebuilt_uuid != second_pending_uuid
    assert rebuilt_uuid != post_second_compaction_uuid

    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    assert cached is not None
    conv = Conversation.model_validate_json(cached)
    assert conv.snapshot_uuid == rebuilt_uuid
    assert conv.ancestor_summaries == ["GEN-1", "GEN-2"]
    assert conv.raw_message_start_index > 0
    tail_contents = [m.content for m in conv.messages]
    assert "after-second-redis-miss" in tail_contents
    assert "restored-again" in tail_contents

    doc = await web_harness.search_client.get_conversation(rebuilt_uuid)
    assert doc is not None
    assert doc["is_compacted"] is False


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_retry_within_recency_tail(web_harness, authed_client):
    """Divergence within the raw window — client edits a user message in the tail (keeps the same source_id, changes
    content). Result: branched snapshot rooted at the parent's shared prefix; ancestor_summaries inherited."""
    conversation_uuid, _pre, post_id, messages = await _establish_compacted_snapshot(web_harness, authed_client)

    # Edit `u4` in place (keep source_id, change content). The reconcile step classifies this as an `edit`, which
    # the web syncer branches by policy.
    edited = [dict(m) for m in messages]
    u4_index = next(i for i, m in enumerate(edited) if m.get("content") == "u4")
    edited[u4_index]["content"] = "u4-edited"
    web_harness.llm_client.set_script(
        LLMScript(rounds=[LLMRound(text_deltas=["a4-fresh"], stop_reason="end_turn", input_tokens=500)])
    )
    record = await post_chat(web_harness, authed_client, conversation_uuid, edited, snapshot_uuid=post_id)
    branched_uuid = record.snapshot_uuid
    assert branched_uuid != post_id

    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    conv = Conversation.model_validate_json(cached)
    assert conv.snapshot_uuid == branched_uuid
    assert conv.parent_snapshot_uuid == post_id
    # Compacted ancestors carry through the branch.
    assert conv.ancestor_summaries == ["STUB SUMMARY"]
    contents = [m.content for m in conv.messages]
    assert "u4-edited" in contents
    assert "a4-fresh" in contents
    # Original `u4` is gone (replaced by `u4-edited`); `a4` survives because its content didn't change and reconcile
    # is source_id-keyed.
    assert "u4" not in contents


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_retry_before_recency_tail(web_harness, authed_client):
    """Divergence before the compaction boundary — the rebuild falls through Case B (no matching compacted ancestor)
    and starts a fresh branch with no ancestors, `raw_message_start_index=0`."""
    conversation_uuid, pre_id, post_id, _messages = await _establish_compacted_snapshot(web_harness, authed_client)

    # Truncate to a single user message — fresh start from the user's POV.
    fresh_messages = [user_message("u1-only")]
    web_harness.llm_client.set_script(
        LLMScript(rounds=[LLMRound(text_deltas=["fresh"], stop_reason="end_turn", input_tokens=500)])
    )
    record = await post_chat(web_harness, authed_client, conversation_uuid, fresh_messages, snapshot_uuid=pre_id)
    branched_uuid = record.snapshot_uuid
    assert branched_uuid != pre_id
    assert branched_uuid != post_id

    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    conv = Conversation.model_validate_json(cached)
    assert conv.snapshot_uuid == branched_uuid
    assert conv.ancestor_summaries == []
    assert conv.raw_message_start_index == 0

    doc = await web_harness.search_client.get_conversation(branched_uuid)
    assert doc is not None
    assert doc["is_compacted"] is False
    assert doc.get("ancestor_summaries") == []
    assert doc.get("raw_message_start_index") == 0
    assert doc.get("parent_snapshot_uuid") is None


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_branch_switch(web_harness, authed_client):
    """After establishing branch A (post-compaction) and branch B (alternate from pre-compaction), the user can
    switch back to branch A by anchoring at A's snapshot. Redis was last written by branch B; branch A is restored
    from ES."""
    conversation_uuid, pre_id, post_id, messages = await _establish_compacted_snapshot(web_harness, authed_client)

    # Branch B: alternate from pre-compaction.
    alternate_messages = [user_message("u1-only")]
    web_harness.llm_client.set_script(
        LLMScript(rounds=[LLMRound(text_deltas=["fresh"], stop_reason="end_turn", input_tokens=500)])
    )
    alt_record = await post_chat(
        web_harness, authed_client, conversation_uuid, alternate_messages, snapshot_uuid=pre_id
    )
    alternate_uuid = alt_record.snapshot_uuid
    assert alternate_uuid != post_id

    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    alt_conv = Conversation.model_validate_json(cached)
    assert alt_conv.snapshot_uuid == alternate_uuid

    # Switch back to branch A by anchoring at post-compaction with the original message list extended.
    messages.append(user_message("back-on-branch-a"))
    web_harness.llm_client.set_script(
        LLMScript(rounds=[LLMRound(text_deltas=["branch-a"], stop_reason="end_turn", input_tokens=500)])
    )
    record = await post_chat(web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=post_id)
    assert record.snapshot_uuid == post_id

    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    conv = Conversation.model_validate_json(cached)
    assert conv.snapshot_uuid == post_id
    assert conv.ancestor_summaries == ["STUB SUMMARY"]
    contents = [m.content for m in conv.messages]
    assert contents[-4:] == ["u4", "a4", "back-on-branch-a", "branch-a"]
