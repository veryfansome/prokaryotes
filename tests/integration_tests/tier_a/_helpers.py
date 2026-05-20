"""Shared helpers for Tier A live-LLM tests.

Unified-conversation wire vocabulary:
- First stream event is a `handshake` carrying `snapshot_uuid`.
- `bot_message` marks final commit with the server-assigned bot `source_id`; subsequent POSTs must echo the
  assistant entry with that `source_id`.
- `/compaction-status` takes `pending_snapshot_uuid`.
- Redis cache parses to `Conversation`.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from uuid import uuid4

from prokaryotes.conversation_v1.models import Conversation
from tests.integration_tests.stream_utils import request_scope

FILLER_PROMPTS = [
    "Write a 200-word paragraph about the history of bread.",
    "Write a 200-word paragraph about ocean currents.",
    "Write a 200-word paragraph about urban planning.",
    "Write a 200-word paragraph about the printing press.",
    "Write a 200-word paragraph about coffee cultivation.",
    "Write a 200-word paragraph about volcanoes.",
]


@dataclass
class TurnRecord:
    """The product of one POST /chat round.

    `bot_message_source_id` is None if the turn aborted before commit (e.g., max-rounds hit, stream error).
    `assistant_text` is accumulated `text_delta`s.
    """

    snapshot_uuid: str
    source_id_assignments: list[dict] = field(default_factory=list)
    bot_message_source_id: str | None = None
    assistant_text: str = ""
    events: list[dict] = field(default_factory=list)


def user_message(content: str) -> dict:
    return {"role": "user", "content": content}


async def post_chat_collect(client, payload: dict) -> TurnRecord:
    """POST /chat, collect the NDJSON stream, and return a TurnRecord."""
    events: list[dict] = []
    async with client.stream("POST", "/chat", json=payload) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.strip():
                events.append(json.loads(line))

    if not events:
        raise AssertionError("Empty stream from /chat")
    handshake = events[0]
    if "snapshot_uuid" not in handshake:
        raise AssertionError(f"First event is not a handshake: {handshake}")
    record = TurnRecord(
        snapshot_uuid=handshake["snapshot_uuid"],
        source_id_assignments=handshake.get("source_id_assignments", []),
        events=events,
    )
    record.assistant_text = "".join(e["text_delta"] for e in events if "text_delta" in e)
    for ev in events:
        if "bot_message" in ev:
            record.bot_message_source_id = ev["bot_message"]["source_id"]
    return record


async def wait_for_compaction(
    client,
    conversation_uuid: str,
    pending_snapshot_uuid: str,
    *,
    attempts: int = 60,
    delay: float = 0.5,
) -> str | None:
    """Poll /compaction-status until done. Returns the relabel target snapshot_uuid (None if none was written —
    e.g., lock released without commit)."""
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


def _apply_assignments(messages: list[dict], assignments: list[dict]) -> None:
    """Stamp server-assigned source_ids onto the local message list via the handshake's client_index → source_id
    map."""
    for assignment in assignments:
        idx = assignment["client_index"]
        sid = assignment["source_id"]
        if 0 <= idx < len(messages):
            messages[idx]["source_id"] = sid


async def drive_to_compaction(
    web_harness,
    authed_client,
    plant: str | None = None,
    *,
    max_turns: int,
    seeded_messages: list[dict] | None = None,
) -> tuple[str, str, str, list[dict]]:
    """Run turns until `compaction_pending` arrives.

    Returns (conversation_uuid, pending_snapshot_uuid, active_snapshot_uuid, messages).
    """
    conversation_uuid = str(uuid4())
    if seeded_messages is not None:
        messages = list(seeded_messages)
    elif plant is not None:
        messages = [user_message(plant)]
    else:
        raise ValueError("drive_to_compaction requires either `plant` or `seeded_messages`")
    snapshot_uuid: str | None = None
    for turn in range(max_turns):
        payload: dict = {"conversation_uuid": conversation_uuid, "messages": messages}
        if snapshot_uuid is not None:
            payload["snapshot_uuid"] = snapshot_uuid
        async with request_scope(web_harness):
            record = await post_chat_collect(authed_client, payload)
        snapshot_uuid = record.snapshot_uuid
        _apply_assignments(messages, record.source_id_assignments)
        if record.assistant_text and record.bot_message_source_id:
            messages.append(
                {
                    "role": "assistant",
                    "content": record.assistant_text,
                    "source_id": record.bot_message_source_id,
                }
            )
        types = [k for ev in record.events for k in ev.keys()]
        if "compaction_pending" in types:
            assert types[-1] == "compaction_pending"
            target = await wait_for_compaction(authed_client, conversation_uuid, snapshot_uuid)
            cached = await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
            assert cached is not None
            active = Conversation.model_validate_json(cached)
            return (
                conversation_uuid,
                snapshot_uuid,
                target or active.snapshot_uuid,
                messages,
            )
        messages.append(user_message(FILLER_PROMPTS[turn % len(FILLER_PROMPTS)]))
    raise AssertionError(f"compaction did not trigger within {max_turns} turns")
