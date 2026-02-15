from types import SimpleNamespace

import pytest

from prokaryotes.anthropic_v1 import LLMClient
from prokaryotes.api_v1.models import (
    ContextPartition,
    ContextPartitionItem,
    ToolParameters,
    ToolSpec,
)


class FakeStreamContext:
    """Mocks the async context manager returned by client.messages.stream()."""

    def __init__(self, text_chunks: list[str], tool_uses: list[dict], stop_reason: str):
        self._text_chunks = text_chunks
        self._tool_uses = tool_uses
        self._stop_reason = stop_reason

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    @property
    def text_stream(self):
        return self._aiter_text()

    async def _aiter_text(self):
        for chunk in self._text_chunks:
            yield chunk

    async def get_final_message(self):
        content = []
        if self._text_chunks:
            content.append(SimpleNamespace(type="text", text="".join(self._text_chunks)))
        for tool_use in self._tool_uses:
            content.append(SimpleNamespace(type="tool_use", **tool_use))
        return SimpleNamespace(content=content, stop_reason=self._stop_reason)


class FakeMessagesAPI:
    def __init__(self, streams):
        self.calls = []
        self._streams = iter(streams)

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return next(self._streams)


class FakeAnthropicClient:
    def __init__(self, streams):
        self.messages = FakeMessagesAPI(streams)
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
                properties={
                    "query": {
                        "type": "string",
                    }
                }
            ),
        )

    async def call(self, arguments: str, call_id: str):
        assert arguments == '{"query":"mars"}'
        assert call_id == "toolu_1"
        return ContextPartitionItem(
            call_id=call_id,
            output="Mars facts",
            type="function_call_output",
        )


@pytest.mark.asyncio
async def test_stream_response_passes_correct_params():
    callback = DummyToolCallback()
    client = LLMClient()
    client.async_anthropic = FakeAnthropicClient([
        FakeStreamContext(text_chunks=["ok"], tool_uses=[], stop_reason="end_turn"),
    ])
    context_partition = ContextPartition(conversation_uuid="test", items=[
        ContextPartitionItem(role="system", content="Be brief"),
        ContextPartitionItem(role="user", content="Hi"),
    ])

    _ = [chunk async for chunk in client.stream_response(
        context_partition=context_partition,
        model="claude-opus-4-7",
        reasoning_effort="medium",
        tool_choice={"type": "any"},
        tool_callbacks={"lookup": callback},
    )]

    assert client.async_anthropic.messages.calls[0] == {
        "model": "claude-opus-4-7",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        "system": "Be brief",
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "tools": [callback.tool_spec.to_anthropic_tool_param()],
        "tool_choice": {"type": "any"},
    }


@pytest.mark.asyncio
async def test_stream_response_yields_text_and_continues_after_tool_callback():
    callback = DummyToolCallback()
    client = LLMClient()
    client.async_anthropic = FakeAnthropicClient([
        FakeStreamContext(
            text_chunks=["Checking "],
            tool_uses=[{"id": "toolu_1", "name": "lookup", "input": {"query": "mars"}}],
            stop_reason="tool_use",
        ),
        FakeStreamContext(
            text_chunks=["Done."],
            tool_uses=[],
            stop_reason="end_turn",
        ),
    ])
    context_partition = ContextPartition(
        conversation_uuid="test",
        items=[ContextPartitionItem(role="user", content="Tell me about Mars")],
    )

    chunks = [
        chunk async for chunk in client.stream_response(
            context_partition=context_partition,
            model="claude-opus-4-7",
            tool_callbacks={"lookup": callback},
        )
    ]

    assert chunks == ["Checking ", "\n", "Done."]
    assert context_partition.items == [
        ContextPartitionItem(role="user", content="Tell me about Mars"),
        ContextPartitionItem(role="assistant", content="Checking "),
        ContextPartitionItem(
            id="toolu_1", call_id="toolu_1", name="lookup",
            arguments='{"query":"mars"}', type="function_call", status="completed",
        ),
        ContextPartitionItem(call_id="toolu_1", output="Mars facts", type="function_call_output"),
        ContextPartitionItem(role="assistant", content="Done."),
    ]

    assert len(client.async_anthropic.messages.calls) == 2
    assert client.async_anthropic.messages.calls[1]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Tell me about Mars"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking "},
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"query": "mars"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Mars facts"}],
        },
    ]
