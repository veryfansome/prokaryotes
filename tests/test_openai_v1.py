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
    assert context_partition.items == [
        ContextPartitionItem(role="user", content="Tell me about Mars"),
        ContextPartitionItem(role="assistant", content="Checking "),
        ContextPartitionItem(
            type="function_call", name="lookup",
            arguments='{"query":"mars"}', call_id="call_1", id="call_1",
        ),
        ContextPartitionItem(call_id="call_1", output="Mars facts", type="function_call_output"),
        ContextPartitionItem(role="assistant", content="Done."),
    ]

    assert len(client.async_openai.responses.calls) == 2
    assert client.async_openai.responses.calls[1]["input"] == [
        {"content": "Tell me about Mars", "role": "user", "type": "message"},
        {"content": "Checking ", "role": "assistant", "type": "message"},
        {"type": "function_call", "name": "lookup", "arguments": '{"query":"mars"}',
         "call_id": "call_1", "id": "call_1"},
        {"type": "function_call_output", "call_id": "call_1", "output": "Mars facts"},
    ]
