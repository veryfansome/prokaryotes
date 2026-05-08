"""Tier B scenarios 1–3: single-turn happy path, multi-turn continuation, tool-call round-trip."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from prokaryotes.api_v1.models import ContextPartition
from tests.integration_tests.fakes import LLMRound, LLMScript, ToolCallSpec
from tests.integration_tests.stream_utils import collect_stream, request_scope

pytestmark = pytest.mark.integration


def _event_types(events: list[dict]) -> list[str]:
    types = []
    for ev in events:
        types.extend(ev.keys())
    return types


def _user_message(content: str) -> dict:
    return {"role": "user", "content": content}


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_single_turn_happy_path(web_harness, authed_client):
    web_harness.llm_client.set_script(
        LLMScript(rounds=[LLMRound(text_deltas=["Hel", "lo!"], stop_reason="end_turn")])
    )
    conversation_uuid = str(uuid4())
    payload = {
        "conversation_uuid": conversation_uuid,
        "messages": [_user_message("Hi")],
    }

    async with request_scope(web_harness):
        async with authed_client.stream("POST", "/chat", json=payload) as response:
            assert response.status_code == 200
            events = await collect_stream(response)

    types = _event_types(events)
    assert types[0] == "partition_uuid"
    ctx_idx = types.index("context_pct")
    first_delta_idx = types.index("text_delta")
    assert ctx_idx < first_delta_idx
    assert "compaction_pending" not in types
    text_concat = "".join(e["text_delta"] for e in events if "text_delta" in e)
    assert text_concat == "Hello!"

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    assert cached is not None
    partition = ContextPartition.model_validate_json(cached)
    assert partition.partition_uuid == events[0]["partition_uuid"]
    # Roles in items: user message + assistant message; system was popped on finalize.
    roles = [item.role for item in partition.items if item.type == "message"]
    assert roles == ["user", "assistant"]

    doc = await web_harness.search_client.get_partition(partition.partition_uuid)
    assert doc is not None
    assert doc["is_compacted"] is False


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_multi_turn_continuation(web_harness, authed_client):
    conversation_uuid = str(uuid4())

    web_harness.llm_client.set_script(
        LLMScript(rounds=[LLMRound(text_deltas=["one"], stop_reason="end_turn")])
    )
    payload_1 = {"conversation_uuid": conversation_uuid, "messages": [_user_message("first")]}
    async with request_scope(web_harness):
        async with authed_client.stream("POST", "/chat", json=payload_1) as response:
            events_1 = await collect_stream(response)
    partition_uuid_1 = events_1[0]["partition_uuid"]
    assistant_1 = "".join(e["text_delta"] for e in events_1 if "text_delta" in e)

    web_harness.llm_client.set_script(
        LLMScript(rounds=[LLMRound(text_deltas=["two"], stop_reason="end_turn")])
    )
    payload_2 = {
        "conversation_uuid": conversation_uuid,
        "partition_uuid": partition_uuid_1,
        "messages": [
            _user_message("first"),
            {"role": "assistant", "content": assistant_1},
            _user_message("second"),
        ],
    }
    with patch.object(
        web_harness.search_client,
        "get_partition",
        AsyncMock(side_effect=AssertionError("Redis fast path should satisfy continuation")),
    ):
        async with request_scope(web_harness):
            async with authed_client.stream("POST", "/chat", json=payload_2) as response:
                events_2 = await collect_stream(response)

    assert events_2[0]["partition_uuid"] == partition_uuid_1

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    partition = ContextPartition.model_validate_json(cached)
    roles = [item.role for item in partition.items if item.type == "message"]
    assert roles == ["user", "assistant", "user", "assistant"]
    contents = [item.content for item in partition.items if item.type == "message"]
    assert contents[0] == "first"
    assert contents[2] == "second"


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_tool_call_round_trip(web_harness, authed_client, request):
    provider = request.node.callspec.params["web_harness"]
    web_harness.llm_client.set_script(
        LLMScript(
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
            ]
        )
    )
    conversation_uuid = str(uuid4())
    payload = {
        "conversation_uuid": conversation_uuid,
        "messages": [_user_message("run a command")],
    }

    async with request_scope(web_harness):
        async with authed_client.stream("POST", "/chat", json=payload) as response:
            events = await collect_stream(response)

    types = _event_types(events)
    assert types[0] == "partition_uuid"
    if provider == "anthropic":
        # Round 1: context_pct → progress_message → tool_call
        ctx_idx = types.index("context_pct")
        assert types[ctx_idx + 1] == "progress_message"
        assert types[ctx_idx + 2] == "tool_call"
    else:
        # OpenAI round 1: tool_call → context_pct → progress_message
        tc_idx = types.index("tool_call")
        assert types[tc_idx + 1] == "context_pct"
        assert types[tc_idx + 2] == "progress_message"
    assert "text_delta" in types

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    partition = ContextPartition.model_validate_json(cached)
    item_types = [item.type for item in partition.items]
    assert "function_call" in item_types
    assert "function_call_output" in item_types
    assert item_types.index("function_call") < item_types.index("function_call_output")
    function_call_item = next(item for item in partition.items if item.type == "function_call")
    assert function_call_item.arguments == json.dumps({"command": "echo hi", "reason": "test"})


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_think_tool_round_trip(web_harness, authed_client):
    web_harness.llm_client.set_script(
        LLMScript(
            rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Let me reason this through."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=json.dumps(
                                {
                                    "context": "Observed facts:\n- The repository contains integration tests.",
                                    "goal": "Decide whether to inspect the integration tests next.",
                                    "perspectives": ["trade-offs", "risks"],
                                }
                            ),
                            call_id="call-think-1",
                            name="think",
                        ),
                    ],
                ),
                LLMRound(text_deltas=["I have a plan."], stop_reason="end_turn"),
            ],
            think_text="STUB THINK ANALYSIS",
        )
    )
    conversation_uuid = str(uuid4())
    payload = {
        "conversation_uuid": conversation_uuid,
        "messages": [_user_message("Think through the next step before answering.")],
    }

    async with request_scope(web_harness):
        async with authed_client.stream("POST", "/chat", json=payload) as response:
            events = await collect_stream(response)

    types = _event_types(events)
    assert types[0] == "partition_uuid"
    assert "tool_call" in types
    assert "text_delta" in types

    cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
    partition = ContextPartition.model_validate_json(cached)
    think_call = next(item for item in partition.items if item.type == "function_call")
    think_output = next(item for item in partition.items if item.type == "function_call_output")
    assert think_call.name == "think"
    assert think_output.call_id == think_call.call_id
    assert think_output.output == "STUB THINK ANALYSIS"
