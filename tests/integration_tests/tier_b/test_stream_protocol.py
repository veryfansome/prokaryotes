"""Tier B scenario 9: NDJSON event-type ordering across providers.

The chat-flow and compaction-flow tests already check provider-specific event order
inside individual scenarios; this file pins down the cross-cutting invariants
(`partition_uuid` first; tool-round order matches the provider contract;
`compaction_pending` is terminal when present; final-round `context_pct`
precedes any `text_delta`).
"""
from __future__ import annotations

import json
from uuid import uuid4

import pytest

from tests.integration_tests.fakes import LLMRound, LLMScript, ToolCallSpec
from tests.integration_tests.stream_utils import collect_stream, request_scope

pytestmark = pytest.mark.integration


def _event_types(events: list[dict]) -> list[str]:
    types = []
    for ev in events:
        types.extend(ev.keys())
    return types


async def _post_events(
    web_harness,
    authed_client,
    conversation_uuid: str,
    *,
    messages: list[dict],
    rounds: list[LLMRound],
    partition_uuid: str | None = None,
) -> list[dict]:
    web_harness.llm_client.set_script(LLMScript(rounds=rounds))
    payload: dict = {"conversation_uuid": conversation_uuid, "messages": messages}
    if partition_uuid is not None:
        payload["partition_uuid"] = partition_uuid
    async with request_scope(web_harness):
        async with authed_client.stream("POST", "/chat", json=payload) as response:
            return await collect_stream(response)


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_partition_uuid_arrives_first(web_harness, authed_client):
    web_harness.llm_client.set_script(
        LLMScript(rounds=[LLMRound(text_deltas=["x"], stop_reason="end_turn")])
    )
    payload = {
        "conversation_uuid": str(uuid4()),
        "messages": [{"role": "user", "content": "hi"}],
    }
    async with request_scope(web_harness):
        async with authed_client.stream("POST", "/chat", json=payload) as response:
            events = await collect_stream(response)
    types = _event_types(events)
    assert types[0] == "partition_uuid"


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_final_round_context_pct_precedes_text_deltas(web_harness, authed_client):
    events = await _post_events(
        web_harness,
        authed_client,
        str(uuid4()),
        messages=[{"role": "user", "content": "hi"}],
        rounds=[LLMRound(text_deltas=["a", "b", "c"], stop_reason="end_turn")],
    )
    types = _event_types(events)
    ctx_idx = types.index("context_pct")
    first_delta_idx = types.index("text_delta")
    assert ctx_idx < first_delta_idx


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_tool_round_order_matches_provider(web_harness, authed_client, request):
    conversation_uuid = str(uuid4())
    provider = request.node.callspec.params["web_harness"]
    events = await _post_events(
        web_harness,
        authed_client,
        conversation_uuid,
        messages=[{"role": "user", "content": "run a command"}],
        rounds=[
            LLMRound(
                stop_reason="tool_use",
                text_deltas=["Let me check..."],
                tool_calls=[
                    ToolCallSpec(
                        arguments=json.dumps({"command": "echo hi", "reason": "test"}),
                        call_id="call-1",
                        name="shell_command",
                    ),
                ],
            ),
            LLMRound(text_deltas=["Done."], stop_reason="end_turn"),
        ],
    )
    types = _event_types(events)
    if provider == "anthropic":
        assert types == [
            "partition_uuid",
            "context_pct",
            "progress_message",
            "tool_call",
            "context_pct",
            "text_delta",
        ]
    else:
        assert types == [
            "partition_uuid",
            "tool_call",
            "context_pct",
            "progress_message",
            "context_pct",
            "text_delta",
        ]


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_compaction_pending_is_last(web_harness, authed_client):
    conversation_uuid = str(uuid4())
    messages: list[dict] = []
    partition_uuid: str | None = None

    for i in range(3):
        messages.append({"role": "user", "content": f"u{i + 1}"})
        events = await _post_events(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[LLMRound(text_deltas=[f"a{i + 1}"], stop_reason="end_turn", input_tokens=500)],
        )
        partition_uuid = events[0]["partition_uuid"]
        assistant_text = "".join(e["text_delta"] for e in events if "text_delta" in e)
        messages.append({"role": "assistant", "content": assistant_text})

    messages.append({"role": "user", "content": "u4"})
    events = await _post_events(
        web_harness,
        authed_client,
        conversation_uuid,
        messages=messages,
        partition_uuid=partition_uuid,
        rounds=[LLMRound(text_deltas=["a4"], stop_reason="end_turn", input_tokens=5000)],
    )
    types = _event_types(events)
    assert "compaction_pending" in types
    assert types[-1] == "compaction_pending"
    assert "compaction_pending" not in types[:-1]
