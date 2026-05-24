"""Tier B scenario 9: NDJSON event-type ordering across providers.

Unified-conversation wire:
- First event is a handshake (`{snapshot_uuid, source_id_assignments, ...}`).
- `bot_message` marks final commit with the server-assigned bot `source_id`.
- `compaction_pending`, when present, is the last persistence-relevant event.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from tests.integration_tests.stream_utils import collect_stream, request_scope
from tests.unit_tests._llm_fakes import LLMRound, LLMScript

pytestmark = pytest.mark.integration


def _event_types(events: list[dict]) -> list[str]:
    types = []
    for ev in events:
        types.extend(ev.keys())
    return types


def _is_handshake(event: dict) -> bool:
    return "snapshot_uuid" in event and "source_id_assignments" in event


async def _post_events(
    web_harness,
    authed_client,
    conversation_uuid: str,
    *,
    messages: list[dict],
    rounds: list[LLMRound],
    snapshot_uuid: str | None = None,
) -> list[dict]:
    web_harness.llm_client.set_script(LLMScript(rounds=rounds))
    payload: dict = {"conversation_uuid": conversation_uuid, "messages": messages}
    if snapshot_uuid is not None:
        payload["snapshot_uuid"] = snapshot_uuid
    async with request_scope(web_harness):
        async with authed_client.stream("POST", "/chat", json=payload) as response:
            return await collect_stream(response)


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_handshake_arrives_first(web_harness, authed_client):
    """First NDJSON event is a handshake carrying `snapshot_uuid` and `source_id_assignments`."""
    web_harness.llm_client.set_script(LLMScript(rounds=[LLMRound(text_deltas=["x"], stop_reason="end_turn")]))
    payload = {
        "conversation_uuid": str(uuid4()),
        "messages": [{"role": "user", "content": "hi"}],
    }
    async with request_scope(web_harness):
        async with authed_client.stream("POST", "/chat", json=payload) as response:
            events = await collect_stream(response)
    assert _is_handshake(events[0])


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_bot_message_marks_final_commit(web_harness, authed_client):
    """A successful turn emits exactly one `bot_message` event with a fresh server-assigned `source_id`."""
    events = await _post_events(
        web_harness,
        authed_client,
        str(uuid4()),
        messages=[{"role": "user", "content": "hi"}],
        rounds=[LLMRound(text_deltas=["a", "b", "c"], stop_reason="end_turn")],
    )
    bot_messages = [e for e in events if "bot_message" in e]
    assert len(bot_messages) == 1
    assert "source_id" in bot_messages[0]["bot_message"]


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
async def test_compaction_pending_is_last(web_harness, authed_client):
    """When compaction triggers, `compaction_pending` is the final NDJSON event."""
    conversation_uuid = str(uuid4())
    messages: list[dict] = []
    snapshot_uuid: str | None = None
    bot_source_id: str | None = None

    for i in range(3):
        if bot_source_id is not None:
            # Echo previous assistant with its server-assigned source_id.
            pass
        messages.append({"role": "user", "content": f"u{i + 1}"})
        events = await _post_events(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            snapshot_uuid=snapshot_uuid,
            rounds=[LLMRound(text_deltas=[f"a{i + 1}"], stop_reason="end_turn", input_tokens=500)],
        )
        snapshot_uuid = events[0]["snapshot_uuid"]
        # Apply source_id_assignments from the handshake.
        for a in events[0].get("source_id_assignments", []):
            messages[a["client_index"]]["source_id"] = a["source_id"]
        # Capture the bot's source_id and echo on next round.
        bot_event = next((e for e in events if "bot_message" in e), None)
        if bot_event:
            bot_source_id = bot_event["bot_message"]["source_id"]
            assistant_text = "".join(e["text_delta"] for e in events if "text_delta" in e)
            messages.append({"role": "assistant", "content": assistant_text, "source_id": bot_source_id})

    messages.append({"role": "user", "content": "u4"})
    events = await _post_events(
        web_harness,
        authed_client,
        conversation_uuid,
        messages=messages,
        snapshot_uuid=snapshot_uuid,
        rounds=[LLMRound(text_deltas=["a4"], stop_reason="end_turn", input_tokens=5000)],
    )
    types = _event_types(events)
    assert "compaction_pending" in types
    assert types[-1] == "compaction_pending"
    assert "compaction_pending" not in types[:-1]
