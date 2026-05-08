"""Tier B scenarios 4–7: forced compaction, retry within tail, retry before tail, branch switch."""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from prokaryotes.api_v1.models import ContextPartition
from tests.integration_tests.fakes import LLMRound, LLMScript
from tests.integration_tests.stream_utils import collect_stream, request_scope

pytestmark = pytest.mark.integration


def _user_message(content: str) -> dict:
    return {"role": "user", "content": content}


async def _wait_for_compaction(
    client,
    conversation_uuid: str,
    pending_partition_uuid: str,
    *,
    attempts: int = 30,
    delay: float = 0.1,
) -> None:
    for _ in range(attempts):
        response = await client.get(
            "/compaction-status",
            params={
                "conversation_uuid": conversation_uuid,
                "pending_partition_uuid": pending_partition_uuid,
            },
        )
        if response.json().get("done"):
            return
        await asyncio.sleep(delay)
    raise AssertionError("compaction did not complete within timeout")


async def _run_turn(
    web_harness,
    authed_client,
    conversation_uuid: str,
    *,
    messages: list[dict],
    partition_uuid: str | None,
    rounds: list[LLMRound],
    summary_delay: float = 0.0,
    summary_text: str = "STUB SUMMARY",
) -> tuple[str, str, list[dict]]:
    web_harness.llm_client.set_script(
        LLMScript(
            rounds=rounds,
            summary_delay=summary_delay,
            summary_text=summary_text,
        )
    )
    payload: dict = {"conversation_uuid": conversation_uuid, "messages": messages}
    if partition_uuid is not None:
        payload["partition_uuid"] = partition_uuid
    async with request_scope(web_harness):
        async with authed_client.stream("POST", "/chat", json=payload) as response:
            events = await collect_stream(response)
    new_partition_uuid = events[0]["partition_uuid"]
    assistant_text = "".join(e["text_delta"] for e in events if "text_delta" in e)
    return new_partition_uuid, assistant_text, events


async def _establish_compacted_partition(
    web_harness,
    authed_client,
    *,
    summary_text: str = "STUB SUMMARY",
) -> tuple[str, str, str, list[dict]]:
    """Run 3 establishing turns and a 4th compaction turn.

    Returns (conversation_uuid, pre_compaction_partition_uuid, post_compaction_partition_uuid,
    full_messages).
    """
    conversation_uuid = str(uuid4())
    messages: list[dict] = []
    partition_uuid: str | None = None
    for i in range(3):
        messages.append(_user_message(f"u{i + 1}"))
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[LLMRound(text_deltas=[f"a{i + 1}"], stop_reason="end_turn", input_tokens=500)],
        )
        messages.append({"role": "assistant", "content": assistant_text})

    messages.append(_user_message("u4"))
    pre_compaction_uuid = partition_uuid
    web_harness.llm_client.set_script(
        LLMScript(
            rounds=[
                LLMRound(text_deltas=["a4"], stop_reason="end_turn", input_tokens=5000),
            ],
            summary_text=summary_text,
        )
    )
    payload: dict = {
        "conversation_uuid": conversation_uuid,
        "partition_uuid": partition_uuid,
        "messages": messages,
    }
    async with request_scope(web_harness):
        async with authed_client.stream("POST", "/chat", json=payload) as response:
            events = await collect_stream(response)
    types = [k for ev in events for k in ev.keys()]
    assert "compaction_pending" in types
    assert types[-1] == "compaction_pending"
    pending_partition_uuid = events[0]["partition_uuid"]
    assistant_text = "".join(e["text_delta"] for e in events if "text_delta" in e)
    messages.append({"role": "assistant", "content": assistant_text})

    await _wait_for_compaction(authed_client, conversation_uuid, pending_partition_uuid)

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    partition = ContextPartition.model_validate_json(cached)
    assert partition.partition_uuid != pending_partition_uuid
    return conversation_uuid, pre_compaction_uuid, partition.partition_uuid, messages


async def _establish_two_generation_compacted_partition(
    web_harness,
    authed_client,
) -> tuple[str, str, str, list[dict]]:
    (
        conversation_uuid,
        _pre_compaction_uuid,
        active_partition_uuid,
        messages,
    ) = await _establish_compacted_partition(
        web_harness,
        authed_client,
        summary_text="GEN-1",
    )

    next_partition_uuid, assistant_text, _ = await _run_turn(
        web_harness,
        authed_client,
        conversation_uuid,
        messages=messages + [_user_message("u5")],
        partition_uuid=active_partition_uuid,
        rounds=[LLMRound(text_deltas=["a5"], stop_reason="end_turn", input_tokens=500)],
    )
    assert next_partition_uuid == active_partition_uuid
    messages.extend([
        _user_message("u5"),
        {"role": "assistant", "content": assistant_text},
    ])

    second_pending_uuid, second_assistant_text, events = await _run_turn(
        web_harness,
        authed_client,
        conversation_uuid,
        messages=messages + [_user_message("u6")],
        partition_uuid=next_partition_uuid,
        rounds=[LLMRound(text_deltas=["a6"], stop_reason="end_turn", input_tokens=5000)],
        summary_text="GEN-2",
    )
    types = [k for ev in events for k in ev.keys()]
    assert "compaction_pending" in types
    assert types[-1] == "compaction_pending"

    await _wait_for_compaction(authed_client, conversation_uuid, second_pending_uuid)

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    partition = ContextPartition.model_validate_json(cached)
    assert partition.partition_uuid != second_pending_uuid
    assert partition.parent_partition_uuid == second_pending_uuid
    assert partition.ancestor_summaries == ["GEN-1", "GEN-2"]

    messages.extend([
        _user_message("u6"),
        {"role": "assistant", "content": second_assistant_text},
    ])
    return conversation_uuid, second_pending_uuid, partition.partition_uuid, messages


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_forced_compaction(web_harness, authed_client):
    (
        conversation_uuid,
        pre_compaction_uuid,
        post_compaction_uuid,
        messages,
    ) = await _establish_compacted_partition(web_harness, authed_client)

    parent_doc = await web_harness.search_client.get_partition(pre_compaction_uuid)
    assert parent_doc is not None
    assert parent_doc["is_compacted"] is True
    assert parent_doc["summary"]
    assert parent_doc["boundary_hash"]
    assert parent_doc["tail_hash"]
    assert parent_doc["boundary_message_count"] > 0
    assert parent_doc["boundary_user_count"] > 0

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    partition = ContextPartition.model_validate_json(cached)
    assert partition.partition_uuid == post_compaction_uuid
    assert partition.parent_partition_uuid == pre_compaction_uuid
    assert partition.ancestor_summaries == ["STUB SUMMARY"]
    assert partition.raw_message_start_index > 0

    new_partition_uuid, _, _ = await _run_turn(
        web_harness,
        authed_client,
        conversation_uuid,
        messages=messages + [_user_message("after-compaction")],
        partition_uuid=pre_compaction_uuid,
        rounds=[LLMRound(text_deltas=["ok"], stop_reason="end_turn", input_tokens=500)],
    )
    assert new_partition_uuid == post_compaction_uuid

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    partition = ContextPartition.model_validate_json(cached)
    contents = [item.content for item in partition.items if item.type == "message"]
    assert contents[-4:] == ["u4", "a4", "after-compaction", "ok"]


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_compaction_carries_forward_messages_added_during_summary(
    web_harness,
    authed_client,
):
    conversation_uuid = str(uuid4())
    messages: list[dict] = []
    partition_uuid: str | None = None
    for i in range(3):
        messages.append(_user_message(f"u{i + 1}"))
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[LLMRound(text_deltas=[f"a{i + 1}"], stop_reason="end_turn", input_tokens=500)],
        )
        messages.append({"role": "assistant", "content": assistant_text})

    messages.append(_user_message("u4"))
    web_harness.llm_client.set_script(
        LLMScript(
            rounds=[LLMRound(text_deltas=["a4"], stop_reason="end_turn", input_tokens=5000)],
            summary_delay=0.3,
            summary_text="GEN-1",
        )
    )
    payload = {
        "conversation_uuid": conversation_uuid,
        "partition_uuid": partition_uuid,
        "messages": messages,
    }
    async with authed_client.stream("POST", "/chat", json=payload) as response:
        events = await collect_stream(response)

    types = [k for ev in events for k in ev.keys()]
    assert "compaction_pending" in types
    assert types[-1] == "compaction_pending"
    pending_partition_uuid = events[0]["partition_uuid"]
    assert await web_harness.redis_client.exists(f"compaction_lock:{conversation_uuid}") == 1

    assistant_text = "".join(e["text_delta"] for e in events if "text_delta" in e)
    messages.append({"role": "assistant", "content": assistant_text})

    follow_up_messages = messages + [_user_message("u5")]
    follow_up_partition_uuid, _, follow_up_events = await _run_turn(
        web_harness,
        authed_client,
        conversation_uuid,
        messages=follow_up_messages,
        partition_uuid=pending_partition_uuid,
        rounds=[LLMRound(text_deltas=["a5"], stop_reason="end_turn", input_tokens=500)],
        summary_text="FOLLOW-UP-SHOULD-NOT-LEAK",
    )
    assert follow_up_partition_uuid == pending_partition_uuid
    follow_up_types = [k for ev in follow_up_events for k in ev.keys()]
    assert "compaction_pending" not in follow_up_types

    await _wait_for_compaction(authed_client, conversation_uuid, pending_partition_uuid)

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    partition = ContextPartition.model_validate_json(cached)
    assert partition.partition_uuid != pending_partition_uuid
    assert partition.parent_partition_uuid == pending_partition_uuid
    assert partition.ancestor_summaries == ["GEN-1"]
    contents = [item.content for item in partition.items if item.type == "message"]
    assert contents[-4:] == ["u4", "a4", "u5", "a5"]


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_compaction_accumulates_ancestor_summaries_across_generations(
    web_harness,
    authed_client,
):
    (
        conversation_uuid,
        second_pending_uuid,
        active_partition_uuid,
        _,
    ) = await _establish_two_generation_compacted_partition(web_harness, authed_client)

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    partition = ContextPartition.model_validate_json(cached)
    assert partition.partition_uuid == active_partition_uuid
    assert partition.partition_uuid != second_pending_uuid
    assert partition.parent_partition_uuid == second_pending_uuid
    assert partition.ancestor_summaries == ["GEN-1", "GEN-2"]


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_rebuild_from_compacted_ancestor_after_redis_miss(web_harness, authed_client):
    (
        conversation_uuid,
        pre_compaction_uuid,
        post_compaction_uuid,
        messages,
    ) = await _establish_compacted_partition(web_harness, authed_client)

    await web_harness.redis_client.delete(f"context_partition:{conversation_uuid}")

    rebuilt_partition_uuid, _, _ = await _run_turn(
        web_harness,
        authed_client,
        conversation_uuid,
        messages=messages + [_user_message("after-redis-miss")],
        partition_uuid=pre_compaction_uuid,
        rounds=[LLMRound(text_deltas=["restored"], stop_reason="end_turn", input_tokens=500)],
    )
    assert rebuilt_partition_uuid != pre_compaction_uuid
    assert rebuilt_partition_uuid != post_compaction_uuid

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    assert cached is not None
    partition = ContextPartition.model_validate_json(cached)
    assert partition.partition_uuid == rebuilt_partition_uuid
    assert partition.parent_partition_uuid == pre_compaction_uuid
    assert partition.ancestor_summaries == ["STUB SUMMARY"]
    assert partition.raw_message_start_index > 0
    contents = [item.content for item in partition.items if item.type == "message"]
    assert contents == ["after-redis-miss", "restored"]

    doc = await web_harness.search_client.get_partition(rebuilt_partition_uuid)
    assert doc is not None
    assert doc["is_compacted"] is False


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_rebuild_from_multigeneration_compacted_ancestor_after_redis_miss(
    web_harness,
    authed_client,
):
    (
        conversation_uuid,
        second_pending_uuid,
        post_second_compaction_uuid,
        messages,
    ) = await _establish_two_generation_compacted_partition(web_harness, authed_client)

    await web_harness.redis_client.delete(f"context_partition:{conversation_uuid}")

    rebuilt_partition_uuid, _, _ = await _run_turn(
        web_harness,
        authed_client,
        conversation_uuid,
        messages=messages + [_user_message("after-second-redis-miss")],
        partition_uuid=second_pending_uuid,
        rounds=[LLMRound(text_deltas=["restored-again"], stop_reason="end_turn", input_tokens=500)],
    )
    assert rebuilt_partition_uuid != second_pending_uuid
    assert rebuilt_partition_uuid != post_second_compaction_uuid

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    assert cached is not None
    partition = ContextPartition.model_validate_json(cached)
    assert partition.partition_uuid == rebuilt_partition_uuid
    assert partition.parent_partition_uuid == second_pending_uuid
    assert partition.ancestor_summaries == ["GEN-1", "GEN-2"]
    assert partition.raw_message_start_index > 0
    contents = [item.content for item in partition.items if item.type == "message"]
    assert contents == ["after-second-redis-miss", "restored-again"]

    doc = await web_harness.search_client.get_partition(rebuilt_partition_uuid)
    assert doc is not None
    assert doc["is_compacted"] is False


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_retry_within_recency_tail(web_harness, authed_client):
    (
        conversation_uuid,
        pre_compaction_uuid,
        post_compaction_uuid,
        messages,
    ) = await _establish_compacted_partition(web_harness, authed_client)

    follow_up_messages = messages[:-1] + [
        {"role": "assistant", "content": "a4-edited"},
        _user_message("u5"),
    ]
    new_partition_uuid, _, _ = await _run_turn(
        web_harness,
        authed_client,
        conversation_uuid,
        messages=follow_up_messages,
        partition_uuid=post_compaction_uuid,
        rounds=[LLMRound(text_deltas=["a5"], stop_reason="end_turn", input_tokens=500)],
    )
    assert new_partition_uuid == post_compaction_uuid

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    partition = ContextPartition.model_validate_json(cached)
    assert partition.partition_uuid == post_compaction_uuid
    assert partition.ancestor_summaries == ["STUB SUMMARY"]
    contents = [item.content for item in partition.items if item.type == "message"]
    assert contents[-4:] == ["u4", "a4-edited", "u5", "a5"]


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_retry_before_recency_tail(web_harness, authed_client):
    (
        conversation_uuid,
        pre_compaction_uuid,
        post_compaction_uuid,
        _,
    ) = await _establish_compacted_partition(web_harness, authed_client)

    truncated_messages = [
        _user_message("u1"),
        {"role": "assistant", "content": "a1"},
        _user_message("u2-edited"),
    ]
    new_partition_uuid, _, _ = await _run_turn(
        web_harness,
        authed_client,
        conversation_uuid,
        messages=truncated_messages,
        partition_uuid=pre_compaction_uuid,
        rounds=[LLMRound(text_deltas=["fresh"], stop_reason="end_turn", input_tokens=500)],
    )
    assert new_partition_uuid != pre_compaction_uuid
    assert new_partition_uuid != post_compaction_uuid

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    partition = ContextPartition.model_validate_json(cached)
    assert partition.partition_uuid == new_partition_uuid
    assert partition.ancestor_summaries == []
    assert partition.raw_message_start_index == 0

    doc = await web_harness.search_client.get_partition(new_partition_uuid)
    assert doc is not None
    assert doc["is_compacted"] is False
    assert doc.get("ancestor_summaries") == []
    assert doc.get("raw_message_start_index") == 0
    assert doc.get("parent_partition_uuid") is None


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_branch_switch(web_harness, authed_client):
    (
        conversation_uuid,
        pre_compaction_uuid,
        post_compaction_uuid,
        messages,
    ) = await _establish_compacted_partition(web_harness, authed_client)

    alternate_messages = [
        _user_message("u1"),
        {"role": "assistant", "content": "a1"},
        _user_message("u2-edited"),
    ]
    alternate_partition_uuid, _, _ = await _run_turn(
        web_harness,
        authed_client,
        conversation_uuid,
        messages=alternate_messages,
        partition_uuid=pre_compaction_uuid,
        rounds=[LLMRound(text_deltas=["fresh"], stop_reason="end_turn", input_tokens=500)],
    )
    assert alternate_partition_uuid != post_compaction_uuid

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    alternate_partition = ContextPartition.model_validate_json(cached)
    assert alternate_partition.partition_uuid == alternate_partition_uuid

    branch_messages = messages + [_user_message("back-on-branch-a")]
    switched_partition_uuid, _, _ = await _run_turn(
        web_harness,
        authed_client,
        conversation_uuid,
        messages=branch_messages,
        partition_uuid=post_compaction_uuid,
        rounds=[LLMRound(text_deltas=["branch-a"], stop_reason="end_turn", input_tokens=500)],
    )
    assert switched_partition_uuid == post_compaction_uuid

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    switched_partition = ContextPartition.model_validate_json(cached)
    assert switched_partition.partition_uuid == post_compaction_uuid
    assert switched_partition.ancestor_summaries == ["STUB SUMMARY"]
    contents = [item.content for item in switched_partition.items if item.type == "message"]
    assert contents[-4:] == ["u4", "a4", "back-on-branch-a", "branch-a"]
