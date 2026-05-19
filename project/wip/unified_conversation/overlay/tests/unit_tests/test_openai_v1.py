"""OpenAIClient streaming + transient-narration invariant."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from openai.types.shared_params import Reasoning

from prokaryotes.api_v1.models import ToolParameters, ToolSpec
from prokaryotes.conversation_v1.models import ProjectedItem, TurnItem
from prokaryotes.openai_v1 import OpenAIClient


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


class DummyToolCallback:
    async def call(self, arguments: str, call_id: str) -> TurnItem:
        assert arguments == '{"query":"mars"}'
        assert call_id == "call_1"
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


class FakeResponsesAPI:
    def __init__(self, event_sequences: list[list]):
        self.calls: list[dict] = []
        self._sequences = iter(event_sequences)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return AsyncEventStream(next(self._sequences))


class FakeAsyncOpenAI:
    def __init__(self, event_sequences: list[list]):
        self.responses = FakeResponsesAPI(event_sequences)
        self.closed = False

    async def close(self):
        self.closed = True


def _make_client(event_sequences):
    client = OpenAIClient()
    client.async_openai = FakeAsyncOpenAI(event_sequences)
    return client


def _user_msg(content: str) -> ProjectedItem:
    return ProjectedItem(type="message", role="user", content=content)


def function_call_done(name: str, arguments: str, call_id: str):
    return SimpleNamespace(
        type="response.output_item.done",
        item=SimpleNamespace(type="function_call", name=name, arguments=arguments, call_id=call_id, id=call_id),
    )


def response_completed(input_tokens: int = 1000, output_tokens: int = 50):
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=usage))


def text_delta(delta: str):
    return SimpleNamespace(type="response.output_text.delta", delta=delta)


def text_done(text: str):
    return SimpleNamespace(type="response.output_text.done", text=text)


@pytest.mark.asyncio
async def test_stream_turn_emits_context_pct_for_ndjson_stream():
    client = _make_client(
        [
            [text_delta("ok"), text_done("ok"), response_completed(input_tokens=102_400)],
        ]
    )

    chunks = [
        chunk
        async for chunk in client.stream_turn(
            items=[_user_msg("Hi")],
            instruction=None,
            model="gpt-5.4-mini",
            stream_ndjson=True,
        )
    ]

    # 102_400 / 128_000 == 80%
    assert '{"context_pct": 80}\n' in chunks


@pytest.mark.asyncio
async def test_intermediate_narration_not_committed():
    """Regression guard: on_committed_turn_item must NEVER receive a `message`
    item. Intermediate narration text from a tool-use round is transient; only
    the final assistant text reaches on_final_assistant_message."""
    callback = DummyToolCallback()
    client = _make_client(
        [
            [
                text_delta("Checking "),
                text_done("Checking "),
                function_call_done("lookup", '{"query":"mars"}', "call_1"),
            ],
            [text_delta("Done."), text_done("Done.")],
        ]
    )
    committed: list[TurnItem] = []
    finals: list[str] = []

    _ = [
        chunk
        async for chunk in client.stream_turn(
            items=[_user_msg("Tell me about Mars")],
            instruction=None,
            model="gpt-5.4",
            on_committed_turn_item=committed.append,
            on_final_assistant_message=finals.append,
            tool_callbacks={"lookup": callback},
        )
    ]

    assert all(item.type in {"function_call", "function_call_output"} for item in committed)
    assert [item.type for item in committed] == ["function_call", "function_call_output"]
    assert finals == ["Done."]


@pytest.mark.asyncio
async def test_stream_turn_passes_correct_params():
    """Provider request is built from `items` + `instruction` + `tool_callbacks`.
    `system` role projected items become `developer` role on the wire."""
    callback = DummyToolCallback()
    client = _make_client(
        [
            [text_delta("ok"), text_done("ok")],
        ]
    )

    _ = [
        chunk
        async for chunk in client.stream_turn(
            items=[_user_msg("Hi")],
            instruction=None,
            model="gpt-5.4",
            reasoning_effort="low",
            tool_callbacks={"lookup": callback},
        )
    ]

    call = client.async_openai.responses.calls[0]
    assert call["model"] == "gpt-5.4"
    assert call["stream"] is True
    assert call["tools"] == [callback.tool_spec.to_openai_function_tool_param()]
    assert call["reasoning"] == Reasoning(effort="low")
    assert call["input"] == [{"content": "Hi", "role": "user", "type": "message"}]


@pytest.mark.asyncio
async def test_stream_turn_yields_text_and_continues_after_tool_callback():
    callback = DummyToolCallback()
    client = _make_client(
        [
            [
                text_delta("Checking "),
                text_done("Checking "),
                function_call_done("lookup", '{"query":"mars"}', "call_1"),
            ],
            [text_delta("Done."), text_done("Done.")],
        ]
    )
    finals: list[str] = []

    chunks = [
        chunk
        async for chunk in client.stream_turn(
            items=[_user_msg("Tell me about Mars")],
            instruction=None,
            model="gpt-5.4",
            on_final_assistant_message=finals.append,
            tool_callbacks={"lookup": callback},
        )
    ]

    assert chunks == ["Checking ", "\n", "Done."]
    assert finals == ["Done."]

    assert len(client.async_openai.responses.calls) == 2
    # Second call sees the projected user message, the dispatched function_call,
    # and the function_call_output; no transient narration leaks back in.
    assert client.async_openai.responses.calls[1]["input"] == [
        {"content": "Tell me about Mars", "role": "user", "type": "message"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"query":"mars"}',
            "status": "completed",
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "Mars facts"},
    ]


@pytest.mark.asyncio
async def test_stream_turn_emits_progress_message_for_ndjson_tool_rounds():
    callback = DummyToolCallback()
    client = _make_client(
        [
            [
                text_delta("Checking "),
                text_done("Checking "),
                function_call_done("lookup", '{"query":"mars"}', "call_1"),
                response_completed(),
            ],
            [
                text_delta("Done."),
                text_done("Done."),
                response_completed(),
            ],
        ]
    )
    committed: list[TurnItem] = []
    finals: list[str] = []

    chunks = [
        chunk
        async for chunk in client.stream_turn(
            items=[_user_msg("Tell me about Mars")],
            instruction=None,
            model="gpt-5.4",
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
    assert all(item.type in {"function_call", "function_call_output"} for item in committed)
    assert finals == ["Done."]
