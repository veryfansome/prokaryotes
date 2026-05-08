"""Shared helpers for Tier A live-LLM tests."""
from __future__ import annotations

import asyncio
from uuid import uuid4

from prokaryotes.api_v1.models import ContextPartition
from tests.integration_tests.stream_utils import request_scope

FILLER_PROMPTS = [
    "Write a 200-word paragraph about the history of bread.",
    "Write a 200-word paragraph about ocean currents.",
    "Write a 200-word paragraph about urban planning.",
    "Write a 200-word paragraph about the printing press.",
    "Write a 200-word paragraph about coffee cultivation.",
    "Write a 200-word paragraph about volcanoes.",
]


def user_message(content: str) -> dict:
    return {"role": "user", "content": content}


async def post_chat_collect(client, payload: dict) -> tuple[list[dict], str, str | None]:
    """POST /chat, return (events, partition_uuid, assistant_text)."""
    import json

    events: list[dict] = []
    async with client.stream("POST", "/chat", json=payload) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.strip():
                events.append(json.loads(line))
    partition_uuid = events[0]["partition_uuid"]
    assistant_text = "".join(e["text_delta"] for e in events if "text_delta" in e)
    return events, partition_uuid, assistant_text


async def wait_for_compaction(
    client,
    conversation_uuid: str,
    pending_partition_uuid: str,
    *,
    attempts: int = 60,
    delay: float = 0.5,
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


async def drive_to_compaction(
    web_harness,
    authed_client,
    plant: str | None = None,
    *,
    max_turns: int,
    seeded_messages: list[dict] | None = None,
) -> tuple[str, str, str, list[dict]]:
    """Run turns until compaction_pending arrives.

    Returns (conversation_uuid, pending_partition_uuid, active_partition_uuid, messages).
    """
    conversation_uuid = str(uuid4())
    if seeded_messages is not None:
        messages = list(seeded_messages)
    elif plant is not None:
        messages = [user_message(plant)]
    else:
        raise ValueError("drive_to_compaction requires either `plant` or `seeded_messages`")
    partition_uuid: str | None = None
    for turn in range(max_turns):
        payload: dict = {"conversation_uuid": conversation_uuid, "messages": messages}
        if partition_uuid is not None:
            payload["partition_uuid"] = partition_uuid
        async with request_scope(web_harness):
            events, partition_uuid, assistant_text = await post_chat_collect(authed_client, payload)
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
        types = [k for ev in events for k in ev.keys()]
        if "compaction_pending" in types:
            assert types[-1] == "compaction_pending"
            await wait_for_compaction(authed_client, conversation_uuid, partition_uuid)
            cached = await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
            assert cached is not None
            active_partition = ContextPartition.model_validate_json(cached)
            return (
                conversation_uuid,
                partition_uuid,
                active_partition.partition_uuid,
                messages,
            )
        messages.append(user_message(FILLER_PROMPTS[turn % len(FILLER_PROMPTS)]))
    raise AssertionError(f"compaction did not trigger within {max_turns} turns")
