"""AnthropicClient streaming + transient-narration invariant."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from prokaryotes.anthropic_v1 import AnthropicClient
from prokaryotes.api_v1.models import ToolParameters, ToolSpec
from prokaryotes.conversation_v1.models import ProjectedItem, TurnItem
from prokaryotes.tools_v1.file_tool import FileTool


class DummyToolCallback:
    """Tool that records its invocation and returns a function_call_output TurnItem."""

    async def call(self, arguments: str, call_id: str) -> TurnItem:
        assert arguments == '{"query":"mars"}'
        assert call_id == "toolu_1"
        return TurnItem(
            call_id=call_id,
            output="Mars facts",
            type="function_call_output",
        )

    @property
    def tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name="lookup",
            description="Look up a record",
            parameters=ToolParameters(properties={"query": {"type": "string"}}),
        )


class FakeStreamContext:
    """Mocks the async context manager returned by client.messages.stream()."""

    def __init__(
        self,
        text_chunks: list[str],
        tool_uses: list[dict],
        stop_reason: str,
        input_tokens: int = 1000,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ):
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


class FakeAsyncAnthropic:
    def __init__(self, streams):
        self.messages = FakeMessagesAPI(streams)
        self.closed = False

    async def close(self):
        self.closed = True


def _make_client(streams):
    client = AnthropicClient()
    client.async_anthropic = FakeAsyncAnthropic(streams)
    return client


def _user_msg(content: str) -> ProjectedItem:
    return ProjectedItem(type="message", role="user", content=content)


@pytest.mark.asyncio
async def test_stream_turn_context_pct_includes_cached_tokens():
    """Total context_pct must sum input + cache_read + cache_creation tokens."""
    client = _make_client(
        [
            FakeStreamContext(
                text_chunks=["ok"],
                tool_uses=[],
                stop_reason="end_turn",
                input_tokens=20_000,
                cache_read_input_tokens=100_000,
                cache_creation_input_tokens=40_000,
            ),
        ]
    )

    chunks = [
        chunk
        async for chunk in client.stream_turn(
            items=[_user_msg("Hi")],
            instruction=None,
            model="claude-opus-4-7",
            stream_ndjson=True,
        )
    ]

    # 20_000 + 100_000 + 40_000 = 160_000; 160_000 / 200_000 == 80%
    assert '{"context_pct": 80}\n' in chunks


@pytest.mark.asyncio
async def test_stream_turn_emits_context_pct_for_ndjson_stream():
    client = _make_client(
        [
            FakeStreamContext(text_chunks=["ok"], tool_uses=[], stop_reason="end_turn", input_tokens=160_000),
        ]
    )

    chunks = [
        chunk
        async for chunk in client.stream_turn(
            items=[_user_msg("Hi")],
            instruction=None,
            model="claude-opus-4-7",
            stream_ndjson=True,
        )
    ]

    assert '{"context_pct": 80}\n' in chunks


@pytest.mark.asyncio
async def test_intermediate_narration_not_committed():
    """Regression guard: on_committed_turn_item must NEVER receive a `message` item. Intermediate narration in a
    tool-use round is transient; only the final assistant text reaches on_final_assistant_message."""
    callback = DummyToolCallback()
    client = _make_client(
        [
            FakeStreamContext(
                text_chunks=["Checking "],
                tool_uses=[{"id": "toolu_1", "name": "lookup", "input": {"query": "mars"}}],
                stop_reason="tool_use",
            ),
            FakeStreamContext(text_chunks=["Done."], tool_uses=[], stop_reason="end_turn"),
        ]
    )
    committed: list[TurnItem] = []
    finals: list[str] = []

    _ = [
        chunk
        async for chunk in client.stream_turn(
            items=[_user_msg("Tell me about Mars")],
            instruction=None,
            model="claude-opus-4-7",
            on_committed_turn_item=committed.append,
            on_final_assistant_message=finals.append,
            tool_callbacks={"lookup": callback},
        )
    ]

    # Committed items: exactly the function_call + function_call_output. No `message`.
    assert all(item.type in {"function_call", "function_call_output"} for item in committed)
    assert [item.type for item in committed] == ["function_call", "function_call_output"]
    # Final assistant text is the second round only, not the intermediate narration.
    assert finals == ["Done."]


@pytest.mark.asyncio
async def test_stream_turn_yields_text_and_continues_after_tool_callback():
    """Tool-use round → final-text round: chunk sequence is correct and on_final_assistant_message sees only the
    final round's text."""
    callback = DummyToolCallback()
    client = _make_client(
        [
            FakeStreamContext(
                text_chunks=["Checking "],
                tool_uses=[{"id": "toolu_1", "name": "lookup", "input": {"query": "mars"}}],
                stop_reason="tool_use",
            ),
            FakeStreamContext(text_chunks=["Done."], tool_uses=[], stop_reason="end_turn"),
        ]
    )
    finals: list[str] = []

    chunks = [
        chunk
        async for chunk in client.stream_turn(
            items=[_user_msg("Tell me about Mars")],
            instruction=None,
            model="claude-opus-4-7",
            on_final_assistant_message=finals.append,
            tool_callbacks={"lookup": callback},
        )
    ]

    assert chunks == ["Checking ", "\n", "Done."]
    assert finals == ["Done."]

    # The second call (after tool dispatch) sees the projected user message, then the assistant tool_use block,
    # then the tool_result. No transient narration text.
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


@pytest.mark.asyncio
async def test_stream_turn_emits_progress_message_for_ndjson_tool_rounds():
    """Mid-tool-use narration emits as progress_message NDJSON (not text_delta) and does NOT call
    on_committed_turn_item with a message item."""
    callback = DummyToolCallback()
    client = _make_client(
        [
            FakeStreamContext(
                text_chunks=["Checking "],
                tool_uses=[{"id": "toolu_1", "name": "lookup", "input": {"query": "mars"}}],
                stop_reason="tool_use",
            ),
            FakeStreamContext(text_chunks=["Done."], tool_uses=[], stop_reason="end_turn"),
        ]
    )
    committed: list[TurnItem] = []
    finals: list[str] = []

    chunks = [
        chunk
        async for chunk in client.stream_turn(
            items=[_user_msg("Tell me about Mars")],
            instruction=None,
            model="claude-opus-4-7",
            on_committed_turn_item=committed.append,
            on_final_assistant_message=finals.append,
            stream_ndjson=True,
            tool_callbacks={"lookup": callback},
        )
    ]

    assert '{"progress_message": "Checking "}\n' in chunks
    assert '{"tool_call": {"name": "lookup", "arguments": "{\\"query\\":\\"mars\\"}"}}\n' in chunks
    assert '{"text_delta": "Checking "}\n' not in chunks
    assert '{"text_delta": "Done."}\n' in chunks
    # No `message`-type item ever reaches the committed callback.
    assert all(item.type in {"function_call", "function_call_output"} for item in committed)
    assert finals == ["Done."]


@pytest.mark.asyncio
async def test_stream_turn_passes_correct_params():
    """The provider call is built from `items` + `instruction` + `tool_callbacks` + `reasoning_effort`.
    ProjectedItem(role=system) reaches the LLM via the `instruction` kwarg, not as a wire-format user/assistant
    message."""
    callback = DummyToolCallback()
    client = _make_client(
        [
            FakeStreamContext(text_chunks=["ok"], tool_uses=[], stop_reason="end_turn"),
        ]
    )

    _ = [
        chunk
        async for chunk in client.stream_turn(
            items=[_user_msg("Hi")],
            instruction="Be brief",
            model="claude-opus-4-7",
            reasoning_effort="medium",
            tool_callbacks={"lookup": callback},
        )
    ]

    assert client.async_anthropic.messages.calls[0] == {
        "model": "claude-opus-4-7",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        "system": "Be brief",
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "tools": [callback.tool_spec.to_anthropic_tool_param()],
    }


@pytest.mark.asyncio
async def test_stream_turn_max_tool_call_rounds_commits_round_then_stops():
    """`max_tool_call_rounds` blocks the *next* LLM call, not the current round's work. With max=1: the first
    round dispatches a tool, callbacks complete, and BOTH the function_call and function_call_output commit
    atomically before the loop breaks — no orphan, no abandoned task."""
    callback = DummyToolCallback()
    client = _make_client(
        [
            FakeStreamContext(
                text_chunks=[""],
                tool_uses=[{"id": "toolu_1", "name": "lookup", "input": {"query": "mars"}}],
                stop_reason="tool_use",
            ),
        ]
    )
    committed: list[TurnItem] = []

    async for _ in client.stream_turn(
        items=[_user_msg("Tell me about Mars")],
        instruction=None,
        model="claude-opus-4-7",
        max_tool_call_rounds=1,
        on_committed_turn_item=committed.append,
        tool_callbacks={"lookup": callback},
    ):
        pass

    # Both the function_call and its function_call_output were committed atomically — neither is orphaned.
    assert len(committed) == 2
    assert committed[0].type == "function_call"
    assert committed[0].call_id == "toolu_1"
    assert committed[1].type == "function_call_output"
    assert committed[1].call_id == "toolu_1"
    assert committed[1].output == "Mars facts"
    # Only one LLM call happened (the limit blocked the next).
    assert len(client.async_anthropic.messages.calls) == 1


def test_anthropic_tool_param_strips_integer_minimum_from_file_tool_schema(tmp_path):
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    anthropic_param = tool.tool_spec.to_anthropic_tool_param()

    start_line = anthropic_param["input_schema"]["properties"]["start_line"]
    end_line = anthropic_param["input_schema"]["properties"]["end_line"]
    assert start_line["type"] == ["integer", "null"]
    assert end_line["type"] == ["integer", "null"]
    assert "minimum" not in start_line
    assert "minimum" not in end_line


def test_openai_tool_param_keeps_integer_minimum_in_file_tool_schema(tmp_path):
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    openai_param = tool.tool_spec.to_openai_function_tool_param()

    start_line = openai_param["parameters"]["properties"]["start_line"]
    end_line = openai_param["parameters"]["properties"]["end_line"]
    assert start_line["minimum"] == 1
    assert end_line["minimum"] == 1


def test_items_to_anthropic_messages_skips_empty_text_blocks():
    """An empty-content message item must not become a `{"type":"text","text":""}` block — Anthropic rejects
    empty text blocks."""
    from prokaryotes.anthropic_v1 import _items_to_anthropic_messages

    items = [
        ProjectedItem(type="message", role="user", content=""),
        ProjectedItem(type="message", role="assistant", content="real reply"),
        ProjectedItem(type="function_call", call_id="c1", name="t", arguments="{}"),
        ProjectedItem(type="function_call_output", call_id="c1", output="done"),
    ]

    messages = _items_to_anthropic_messages(items)

    text_blocks = [
        block for message in messages for block in message["content"] if block.get("type") == "text"
    ]
    assert all(block["text"] for block in text_blocks)
    assert any(block["text"] == "real reply" for block in text_blocks)
