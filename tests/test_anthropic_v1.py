from types import SimpleNamespace

import pytest

from prokaryotes.anthropic_v1 import AnthropicClient
from prokaryotes.api_v1.models import (
    ContextPartition,
    ContextPartitionItem,
    ToolParameters,
    ToolSpec,
)


class DummyToolCallback:
    async def call(self, arguments: str, call_id: str):
        assert arguments == '{"query":"mars"}'
        assert call_id == "toolu_1"
        return ContextPartitionItem(
            call_id=call_id,
            output="Mars facts",
            type="function_call_output",
        )

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


class FakeStreamContext:
    """Mocks the async context manager returned by client.messages.stream()."""

    def __init__(self, text_chunks: list[str], tool_uses: list[dict], stop_reason: str,
                 input_tokens: int = 1000, cache_read_input_tokens: int = 0,
                 cache_creation_input_tokens: int = 0):
        self._text_chunks = text_chunks
        self._tool_uses = tool_uses
        self._stop_reason = stop_reason
        self._input_tokens = input_tokens
        self._cache_read_input_tokens = cache_read_input_tokens
        self._cache_creation_input_tokens = cache_creation_input_tokens

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def _aiter_text(self):
        for chunk in self._text_chunks:
            yield chunk

    async def get_final_message(self):
        content = []
        if self._text_chunks:
            content.append(SimpleNamespace(type="text", text="".join(self._text_chunks)))
        for tool_use in self._tool_uses:
            content.append(SimpleNamespace(type="tool_use", **tool_use))
        usage = SimpleNamespace(
            input_tokens=self._input_tokens,
            output_tokens=50,
            cache_read_input_tokens=self._cache_read_input_tokens,
            cache_creation_input_tokens=self._cache_creation_input_tokens,
        )
        return SimpleNamespace(content=content, stop_reason=self._stop_reason, usage=usage)

    @property
    def text_stream(self):
        return self._aiter_text()


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


@pytest.mark.asyncio
async def test_stream_turn_context_pct_includes_cached_tokens():
    # cache_read_input_tokens and cache_creation_input_tokens must be summed with
    # input_tokens; omitting them causes the indicator to reset when the cache warms.
    client = AnthropicClient()
    client.async_anthropic = FakeAnthropicClient([
        FakeStreamContext(
            text_chunks=["ok"], tool_uses=[], stop_reason="end_turn",
            input_tokens=20_000,
            cache_read_input_tokens=100_000,
            cache_creation_input_tokens=40_000,
        ),
    ])
    context_partition = ContextPartition(
        conversation_uuid="test",
        items=[ContextPartitionItem(role="user", content="Hi")],
    )

    chunks = [
        chunk async for chunk in client.stream_turn(
            context_partition=context_partition,
            model="claude-opus-4-7",
            stream_ndjson=True,
        )
    ]

    # total = 20_000 + 100_000 + 40_000 = 160_000; 160_000 / 200_000 * 100 == 80
    assert '{"context_pct": 80}\n' in chunks


@pytest.mark.asyncio
async def test_stream_turn_emits_context_pct_for_ndjson_stream():
    # 160_000 / 200_000 * 100 == 80; validates the percentage math, not just presence.
    client = AnthropicClient()
    client.async_anthropic = FakeAnthropicClient([
        FakeStreamContext(text_chunks=["ok"], tool_uses=[], stop_reason="end_turn", input_tokens=160_000),
    ])
    context_partition = ContextPartition(
        conversation_uuid="test",
        items=[ContextPartitionItem(role="user", content="Hi")],
    )

    chunks = [
        chunk async for chunk in client.stream_turn(
            context_partition=context_partition,
            model="claude-opus-4-7",
            stream_ndjson=True,
        )
    ]

    assert '{"context_pct": 80}\n' in chunks


@pytest.mark.asyncio
async def test_stream_turn_no_intermediate_assistant_item_before_tool_call():
    # Regression guard: the partition must NOT contain a standalone assistant text item
    # positioned before a function_call item. Such a shape causes sync_from_conversation
    # to truncate the context on the next request.
    callback = DummyToolCallback()
    client = AnthropicClient()
    client.async_anthropic = FakeAnthropicClient([
        FakeStreamContext(
            text_chunks=["Checking "],
            tool_uses=[{"id": "toolu_1", "name": "lookup", "input": {"query": "mars"}}],
            stop_reason="tool_use",
        ),
        FakeStreamContext(text_chunks=["Done."], tool_uses=[], stop_reason="end_turn"),
    ])
    context_partition = ContextPartition(
        conversation_uuid="test",
        items=[ContextPartitionItem(role="user", content="Tell me about Mars")],
    )

    _ = [chunk async for chunk in client.stream_turn(
        context_partition=context_partition,
        model="claude-opus-4-7",
        tool_callbacks={"lookup": callback},
    )]

    items = context_partition.items
    function_call_idx = next(i for i, it in enumerate(items) if it.type == "function_call")
    # No standalone assistant message should appear before the function_call item
    assert not any(
        it.type == "message" and it.role == "assistant"
        for it in items[:function_call_idx]
    )


@pytest.mark.asyncio
async def test_stream_turn_passes_correct_params():
    callback = DummyToolCallback()
    client = AnthropicClient()
    client.async_anthropic = FakeAnthropicClient([
        FakeStreamContext(text_chunks=["ok"], tool_uses=[], stop_reason="end_turn"),
    ])
    context_partition = ContextPartition(conversation_uuid="test", items=[
        ContextPartitionItem(role="system", content="Be brief"),
        ContextPartitionItem(role="user", content="Hi"),
    ])

    _ = [chunk async for chunk in client.stream_turn(
        context_partition=context_partition,
        model="claude-opus-4-7",
        reasoning_effort="medium",
        tool_callbacks={"lookup": callback},
    )]

    assert client.async_anthropic.messages.calls[0] == {
        "model": "claude-opus-4-7",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        "system": "Be brief",
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "tools": [callback.tool_spec.to_anthropic_tool_param()],
    }


@pytest.mark.asyncio
async def test_stream_turn_yields_text_and_continues_after_tool_callback():
    callback = DummyToolCallback()
    client = AnthropicClient()
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
        chunk async for chunk in client.stream_turn(
            context_partition=context_partition,
            model="claude-opus-4-7",
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
            id="toolu_1", call_id="toolu_1", name="lookup",
            arguments='{"query":"mars"}', type="function_call", status="completed",
            text_preamble="Checking ",
        ),
        ContextPartitionItem(call_id="toolu_1", output="Mars facts", type="function_call_output"),
        ContextPartitionItem(role="assistant", content="Checking \nDone."),
    ]

    assert len(client.async_anthropic.messages.calls) == 2
    # The second call must still receive the preamble text as a text block in the same
    # assistant turn as the tool_use block — as required by the Anthropic Messages API.
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
