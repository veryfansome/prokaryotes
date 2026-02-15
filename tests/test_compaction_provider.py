import json
from types import SimpleNamespace

import pytest

from prokaryotes.api_v1.models import (
    ContextPartition,
    ContextPartitionItem,
)
from tests.context_partition_utils import make_message_items


def make_anthropic_harness(create_fn):
    from prokaryotes.anthropic_v1.web_harness import WebHarness

    harness = WebHarness.__new__(WebHarness)
    harness.llm_client = SimpleNamespace(
        async_anthropic=SimpleNamespace(
            messages=SimpleNamespace(create=create_fn)
        )
    )
    return harness


def make_openai_harness(create_fn):
    from prokaryotes.openai_v1.web_harness import WebHarness

    harness = WebHarness.__new__(WebHarness)
    harness.llm_client = SimpleNamespace(
        async_openai=SimpleNamespace(
            responses=SimpleNamespace(create=create_fn)
        )
    )
    return harness


@pytest.mark.asyncio
async def test_anthropic_summarize_and_compact_includes_ancestor_summaries_in_system_string():
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text="Summary.")])

    harness = make_anthropic_harness(fake_create)
    snapshot = ContextPartition(
        conversation_uuid="test",
        ancestor_summaries=["S0"],
        items=make_message_items(("user", "U1"), ("assistant", "A1")),
    )

    result = await harness._summarize_and_compact(model="claude-opus-4-7", snapshot=snapshot)

    create_call = calls[0]
    assert result == "Summary."
    assert create_call["system"].startswith("# Compacted conversation summary")
    assert "S0" in create_call["system"]
    assert "S0" not in json.dumps(create_call["messages"])


@pytest.mark.asyncio
async def test_openai_summarize_and_compact_excludes_text_preamble():
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text="Summary.")

    harness = make_openai_harness(fake_create)
    snapshot = ContextPartition(
        conversation_uuid="test",
        items=[
            ContextPartitionItem(role="user", content="Tell me about Mars"),
            ContextPartitionItem(
                type="function_call",
                name="lookup",
                arguments='{"query":"mars"}',
                call_id="call_1",
                id="call_1",
                text_preamble="Let me look that up. ",
            ),
            ContextPartitionItem(call_id="call_1", output="Mars facts", type="function_call_output"),
            ContextPartitionItem(role="assistant", content="Let me look that up. Here are the facts."),
        ],
    )

    result = await harness._summarize_and_compact(model="gpt-5.4-mini", snapshot=snapshot)

    input_items = calls[0]["input"]
    assert result == "Summary."
    for item in input_items:
        assert "text_preamble" not in item

    preamble_msgs = [
        item for item in input_items
        if item.get("role") == "assistant" and item.get("content") == "Let me look that up. "
    ]
    assert len(preamble_msgs) == 1

    preamble_idx = input_items.index(preamble_msgs[0])
    func_call_idx = next(i for i, item in enumerate(input_items) if item.get("type") == "function_call")
    assert preamble_idx + 1 == func_call_idx


@pytest.mark.asyncio
async def test_openai_summarize_and_compact_includes_ancestor_summaries_in_developer_message():
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text="Summary.")

    harness = make_openai_harness(fake_create)
    snapshot = ContextPartition(
        conversation_uuid="test",
        ancestor_summaries=["S0"],
        items=make_message_items(("user", "U1"), ("assistant", "A1")),
    )

    result = await harness._summarize_and_compact(model="gpt-5.4-mini", snapshot=snapshot)

    input_items = calls[0]["input"]
    developer_idx = next(i for i, item in enumerate(input_items) if item.get("role") == "developer")
    first_raw_idx = next(i for i, item in enumerate(input_items) if item.get("role") == "user")

    assert result == "Summary."
    assert input_items[developer_idx]["content"].startswith("# Compacted conversation summary")
    assert "S0" in input_items[developer_idx]["content"]
    assert developer_idx < first_raw_idx


def test_to_anthropic_messages_appends_ancestor_summaries_after_system_instructions():
    context_partition = ContextPartition(
        conversation_uuid="test",
        ancestor_summaries=["Summary 1", "Summary 2"],
        items=[
            ContextPartitionItem(role="system", content="Be concise"),
            ContextPartitionItem(role="user", content="Hello"),
        ],
    )

    system, messages = context_partition.to_anthropic_messages()

    assert system.startswith("Be concise")
    assert system.index("Be concise") < system.index("# Compacted conversation summary")
    assert "Summary 2" in system
    assert "Summary 1" in system
    assert messages == [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    ]
