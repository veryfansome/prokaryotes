from types import SimpleNamespace

import pytest
from openai.types.shared_params import Reasoning

from prokaryotes.api_v1.models import (
    ContextPartition,
    ContextPartitionItem,
    ToolParameters,
    ToolSpec,
)
from prokaryotes.openai_v1 import LLMClient


class AsyncEventStream:
    def __init__(self, events: list):
        self._events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None


class FakeResponsesAPI:
    def __init__(self, event_sequences: list[list]):
        self.calls = []
        self._sequences = iter(event_sequences)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return AsyncEventStream(next(self._sequences))


class FakeOpenAIClient:
    def __init__(self, event_sequences: list[list]):
        self.responses = FakeResponsesAPI(event_sequences)
        self.closed = False

    async def close(self):
        self.closed = True


class DummyToolCallback:
    @property
    def tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name="lookup",
            description="Look up a record",
            parameters=ToolParameters(
                properties={"query": {"type": "string"}},
            ),
        )

    async def call(self, arguments: str, call_id: str):
        assert arguments == '{"query":"mars"}'
        assert call_id == "call_1"
        return ContextPartitionItem(
            call_id=call_id,
            output="Mars facts",
            type="function_call_output",
        )


def function_call_done(name: str, arguments: str, call_id: str):
    return SimpleNamespace(
        type="response.output_item.done",
        item=SimpleNamespace(type="function_call", name=name, arguments=arguments, call_id=call_id, id=call_id),
    )


def text_delta(delta: str):
    return SimpleNamespace(type="response.output_text.delta", delta=delta)


def text_done(text: str):
    return SimpleNamespace(type="response.output_text.done", text=text)


def response_completed(input_tokens: int = 1000, output_tokens: int = 50):
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=usage))


@pytest.mark.asyncio
async def test_stream_response_passes_correct_params():
    callback = DummyToolCallback()
    client = LLMClient()
    client.async_openai = FakeOpenAIClient([
        [text_delta("ok"), text_done("ok")],
    ])
    context_partition = ContextPartition(conversation_uuid="test", items=[
        ContextPartitionItem(role="user", content="Hi"),
    ])

    _ = [chunk async for chunk in client.stream_response(
        context_partition=context_partition,
        model="gpt-5.4",
        reasoning_effort="low",
        tool_callbacks={"lookup": callback},
    )]

    call = client.async_openai.responses.calls[0]
    assert call["model"] == "gpt-5.4"
    assert call["stream"] is True
    assert call["tools"] == [callback.tool_spec.to_openai_function_tool_param()]
    assert call["tool_choice"] == "auto"
    assert call["reasoning"] == Reasoning(effort="low")
    assert call["input"] == [{"content": "Hi", "role": "user", "type": "message"}]


@pytest.mark.asyncio
async def test_stream_response_yields_text_and_continues_after_tool_callback():
    callback = DummyToolCallback()
    client = LLMClient()
    client.async_openai = FakeOpenAIClient([
        [
            text_delta("Checking "),
            text_done("Checking "),
            function_call_done("lookup", '{"query":"mars"}', "call_1"),
        ],
        [
            text_delta("Done."),
            text_done("Done."),
        ],
    ])
    context_partition = ContextPartition(
        conversation_uuid="test",
        items=[ContextPartitionItem(role="user", content="Tell me about Mars")],
    )

    chunks = [
        chunk async for chunk in client.stream_response(
            context_partition=context_partition,
            model="gpt-5.4",
            tool_callbacks={"lookup": callback},
        )
    ]

    assert chunks == ["Checking ", "\n", "Done."]

    # Intermediate text is stored as text_preamble on the function_call item, not as a
    # standalone assistant message. The single combined assistant message at the end
    # matches what the UI accumulates in fullResponse — preventing sync truncation.
    assert context_partition.items == [
        ContextPartitionItem(role="user", content="Tell me about Mars"),
        ContextPartitionItem(
            type="function_call", name="lookup",
            arguments='{"query":"mars"}', call_id="call_1", id="call_1",
            text_preamble="Checking ",
        ),
        ContextPartitionItem(call_id="call_1", output="Mars facts", type="function_call_output"),
        ContextPartitionItem(role="assistant", content="Checking \nDone."),
    ]

    assert len(client.async_openai.responses.calls) == 2
    # The second call must reconstruct the text item from text_preamble so the
    # Responses API receives the original model turn in the correct order.
    assert client.async_openai.responses.calls[1]["input"] == [
        {"content": "Tell me about Mars", "role": "user", "type": "message"},
        {"content": "Checking ", "role": "assistant", "type": "message"},
        {"type": "function_call", "name": "lookup", "arguments": '{"query":"mars"}',
         "call_id": "call_1", "id": "call_1"},
        {"type": "function_call_output", "call_id": "call_1", "output": "Mars facts"},
    ]


@pytest.mark.asyncio
async def test_stream_response_emits_context_pct_for_ndjson_stream():
    # 102_400 / 128_000 * 100 == 80; validates the percentage math, not just presence.
    client = LLMClient()
    client.async_openai = FakeOpenAIClient([
        [text_delta("ok"), text_done("ok"), response_completed(input_tokens=102_400)],
    ])
    context_partition = ContextPartition(
        conversation_uuid="test",
        items=[ContextPartitionItem(role="user", content="Hi")],
    )

    chunks = [
        chunk async for chunk in client.stream_response(
            context_partition=context_partition,
            model="gpt-5.4-mini",
            stream_ndjson=True,
        )
    ]

    assert '{"context_pct": 80}\n' in chunks


@pytest.mark.asyncio
async def test_summarize_and_compact_excludes_text_preamble():
    from prokaryotes.openai_v1.web_harness import WebHarness

    harness = WebHarness.__new__(WebHarness)

    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text="Summary.")

    harness.llm_client = SimpleNamespace(
        async_openai=SimpleNamespace(
            responses=SimpleNamespace(create=fake_create)
        )
    )

    snapshot = ContextPartition(
        conversation_uuid="test",
        items=[
            ContextPartitionItem(role="user", content="Tell me about Mars"),
            ContextPartitionItem(
                type="function_call", name="lookup",
                arguments='{"query":"mars"}', call_id="call_1", id="call_1",
                text_preamble="Let me look that up. ",
            ),
            ContextPartitionItem(call_id="call_1", output="Mars facts", type="function_call_output"),
            ContextPartitionItem(role="assistant", content="Let me look that up. Here are the facts."),
        ],
    )

    result = await harness._summarize_and_compact(snapshot=snapshot, model="gpt-5.4-mini")

    assert result == "Summary."
    assert len(calls) == 1
    input_items = calls[0]["input"]

    # No item in the API payload should carry the internal text_preamble field
    for item in input_items:
        assert "text_preamble" not in item

    # The preamble text must be reconstructed as a preceding assistant message
    preamble_msgs = [
        item for item in input_items
        if item.get("role") == "assistant" and item.get("content") == "Let me look that up. "
    ]
    assert len(preamble_msgs) == 1

    # It must appear immediately before the function_call item
    preamble_idx = input_items.index(preamble_msgs[0])
    func_call_idx = next(i for i, item in enumerate(input_items) if item.get("type") == "function_call")
    assert preamble_idx + 1 == func_call_idx


@pytest.mark.asyncio
async def test_stream_response_no_intermediate_assistant_item_before_tool_call():
    # Regression guard: the partition must NOT contain a standalone assistant text item
    # positioned before a function_call item.
    callback = DummyToolCallback()
    client = LLMClient()
    client.async_openai = FakeOpenAIClient([
        [
            text_delta("Checking "),
            text_done("Checking "),
            function_call_done("lookup", '{"query":"mars"}', "call_1"),
        ],
        [text_delta("Done."), text_done("Done.")],
    ])
    context_partition = ContextPartition(
        conversation_uuid="test",
        items=[ContextPartitionItem(role="user", content="Tell me about Mars")],
    )

    _ = [chunk async for chunk in client.stream_response(
        context_partition=context_partition,
        model="gpt-5.4",
        tool_callbacks={"lookup": callback},
    )]

    items = context_partition.items
    function_call_idx = next(i for i, it in enumerate(items) if it.type == "function_call")
    assert not any(
        it.type == "message" and it.role == "assistant"
        for it in items[:function_call_idx]
    )
