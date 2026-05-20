"""Shared Tier-B helpers for the unified-conversation wire.

Wrap the handshake/bot_message protocol so per-test code stays terse:
- POST /chat → TurnRecord (snapshot_uuid + source_id_assignments + bot_source_id + text + events).
- `apply_assignments` stamps server-assigned source_ids back onto the message list.
- `echo_assistant` appends the assistant entry with its captured source_id so the next POST passes the DAG-scoped
  guardrail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tests.integration_tests.stream_utils import collect_stream, request_scope


@dataclass
class TurnRecord:
    snapshot_uuid: str
    source_id_assignments: list[dict] = field(default_factory=list)
    bot_message_source_id: str | None = None
    assistant_text: str = ""
    events: list[dict] = field(default_factory=list)


def event_types(events: list[dict]) -> list[str]:
    types = []
    for ev in events:
        types.extend(ev.keys())
    return types


def is_handshake(event: dict) -> bool:
    return "snapshot_uuid" in event and "source_id_assignments" in event


def user_message(content: str) -> dict:
    return {"role": "user", "content": content}


def apply_assignments(messages: list[dict], assignments: list[dict]) -> None:
    for a in assignments:
        idx = a["client_index"]
        sid = a["source_id"]
        if 0 <= idx < len(messages):
            messages[idx]["source_id"] = sid


def echo_assistant(messages: list[dict], record: TurnRecord) -> None:
    """Append the assistant entry with its server-assigned source_id so the next POST's DAG-scoped guardrail
    recognizes it."""
    if record.assistant_text and record.bot_message_source_id:
        messages.append(
            {
                "role": "assistant",
                "content": record.assistant_text,
                "source_id": record.bot_message_source_id,
            }
        )


async def post_chat(
    web_harness,
    authed_client,
    conversation_uuid: str,
    messages: list[dict],
    *,
    snapshot_uuid: str | None = None,
) -> TurnRecord:
    payload: dict = {"conversation_uuid": conversation_uuid, "messages": messages}
    if snapshot_uuid is not None:
        payload["snapshot_uuid"] = snapshot_uuid
    async with request_scope(web_harness):
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


async def post_chat_and_advance(
    web_harness,
    authed_client,
    conversation_uuid: str,
    messages: list[dict],
    *,
    snapshot_uuid: str | None = None,
) -> TurnRecord:
    """post_chat + stamp assignments + echo assistant onto `messages` in place."""
    record = await post_chat(web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid)
    apply_assignments(messages, record.source_id_assignments)
    echo_assistant(messages, record)
    return record
