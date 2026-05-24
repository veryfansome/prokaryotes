"""Tier B scenarios 1–3: single-turn happy path, multi-turn continuation, tool-call round-trip.

Unified-conversation wire: handshake first event, bot_message final commit, snapshot_uuid + source_id_assignments.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from prokaryotes.conversation_v1.models import Conversation
from tests.integration_tests.tier_b._helpers import (
    event_types,
    is_handshake,
    post_chat,
    post_chat_and_advance,
    user_message,
)
from tests.unit_tests._llm_fakes import LLMRound, LLMScript, ToolCallSpec

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_single_turn_happy_path(web_harness, authed_client):
    web_harness.llm_client.set_script(LLMScript(rounds=[LLMRound(text_deltas=["Hel", "lo!"], stop_reason="end_turn")]))
    conversation_uuid = str(uuid4())

    record = await post_chat(web_harness, authed_client, conversation_uuid, [user_message("Hi")])

    types = event_types(record.events)
    assert is_handshake(record.events[0])
    ctx_idx = types.index("context_pct")
    first_delta_idx = types.index("text_delta")
    assert ctx_idx < first_delta_idx
    assert "compaction_pending" not in types
    assert record.assistant_text == "Hello!"
    assert record.bot_message_source_id is not None

    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    conv = Conversation.model_validate_json(cached)
    assert conv.snapshot_uuid == record.snapshot_uuid
    author_ids = [m.author_id for m in conv.messages]
    assert author_ids[0] != conv.bot_author_id  # user first
    assert author_ids[-1] == conv.bot_author_id  # bot last

    doc = await web_harness.search_client.get_conversation(conv.snapshot_uuid)
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

    web_harness.llm_client.set_script(LLMScript(rounds=[LLMRound(text_deltas=["one"], stop_reason="end_turn")]))
    messages: list[dict] = [user_message("first")]
    record_1 = await post_chat_and_advance(web_harness, authed_client, conversation_uuid, messages)
    snapshot_uuid_1 = record_1.snapshot_uuid

    web_harness.llm_client.set_script(LLMScript(rounds=[LLMRound(text_deltas=["two"], stop_reason="end_turn")]))
    messages.append(user_message("second"))
    with patch.object(
        web_harness.search_client,
        "get_conversation",
        AsyncMock(side_effect=AssertionError("Redis fast path should satisfy continuation")),
    ):
        record_2 = await post_chat(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid_1
        )

    assert record_2.snapshot_uuid == snapshot_uuid_1

    cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
    conv = Conversation.model_validate_json(cached)
    author_ids = [m.author_id for m in conv.messages]
    # user, bot, user, bot
    assert len(author_ids) == 4
    assert author_ids[0] != conv.bot_author_id
    assert author_ids[1] == conv.bot_author_id
    assert author_ids[2] != conv.bot_author_id
    assert author_ids[3] == conv.bot_author_id
    contents = [m.content for m in conv.messages]
    assert contents[0] == "first"
    assert contents[2] == "second"


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_tool_call_round_trip(web_harness, authed_client):
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
    record = await post_chat(web_harness, authed_client, conversation_uuid, [user_message("run a command")])

    # The exact interleaving of context_pct/progress_message/tool_call differs by provider and is intentionally
    # not pinned — only their presence is part of the contract.
    types_after = event_types(record.events[1:])
    assert "tool_call" in types_after
    assert "text_delta" in types_after
    assert record.bot_message_source_id is not None

    # TurnExecution carries both function_call and function_call_output.
    te = await web_harness.search_client.get_turn_execution(conversation_uuid, record.bot_message_source_id)
    assert te is not None
    item_types = [item.type for item in te.items]
    assert "function_call" in item_types
    assert "function_call_output" in item_types
    assert item_types.index("function_call") < item_types.index("function_call_output")
    function_call_item = next(item for item in te.items if item.type == "function_call")
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
    record = await post_chat(
        web_harness,
        authed_client,
        conversation_uuid,
        [user_message("Think through the next step before answering.")],
    )

    types = event_types(record.events)
    assert is_handshake(record.events[0])
    assert "tool_call" in types
    assert "text_delta" in types

    te = await web_harness.search_client.get_turn_execution(conversation_uuid, record.bot_message_source_id)
    assert te is not None
    think_call = next(item for item in te.items if item.type == "function_call")
    think_output = next(item for item in te.items if item.type == "function_call_output")
    assert think_call.name == "think"
    assert think_output.call_id == think_call.call_id
    # The think tool calls llm_client.complete, which returns think_text.
    assert think_output.output == "STUB THINK ANALYSIS"


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_post_rejects_unknown_assistant_source_id(web_harness, authed_client):
    """Issue 2 guardrail: assistant entries with unknown source_id are 400-rejected."""
    web_harness.llm_client.set_script(LLMScript(rounds=[LLMRound(text_deltas=["x"], stop_reason="end_turn")]))
    conversation_uuid = str(uuid4())
    payload = {
        "conversation_uuid": conversation_uuid,
        "messages": [
            user_message("first"),
            {"role": "assistant", "content": "fabricated reply", "source_id": "fake-id-99"},
        ],
    }
    response = await authed_client.post("/chat", json=payload)
    assert response.status_code == 400
