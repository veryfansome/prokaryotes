"""Scriptable fake LLM clients for Tier B.

The fakes substitute at the harness-contract level: `stream_turn` emits NDJSON directly
rather than mimicking provider streaming SDK events. They retain provider-specific event
ordering for tool-call rounds and reproduce every side effect the harness depends on
(on_usage callback, ContextPartition mutation, tool_callback dispatch).
"""
from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from prokaryotes.api_v1.models import (
    ContextPartition,
    ContextPartitionItem,
    FunctionToolCallback,
)
from prokaryotes.utils_v1.llm_utils import (
    DEFAULT_CONTEXT_WINDOW,
    MODEL_CONTEXT_WINDOWS,
)


@dataclass
class ToolCallSpec:
    arguments: str
    call_id: str
    name: str


@dataclass
class LLMRound:
    input_tokens: int = 1000
    output_tokens: int = 200
    stop_reason: str = "end_turn"
    text_deltas: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallSpec] = field(default_factory=list)


@dataclass
class LLMScript:
    rounds: list[LLMRound]
    summary_delay: float = 0.0
    summary_text: str = "STUB SUMMARY"
    think_text: str = "STUB THINK ANALYSIS"


class FakeAnthropicClient:
    def __init__(self) -> None:
        self._round_cursor = 0
        self._script: LLMScript | None = None
        self.summary_create_calls: list[dict[str, Any]] = []
        self.stream_context_partitions: list[ContextPartition] = []
        self.async_anthropic = SimpleNamespace(
            messages=SimpleNamespace(create=self._messages_create),
        )

    async def close(self) -> None:
        pass

    async def complete(
        self,
        context_partition: ContextPartition,
        model: str,
        reasoning_effort: str | None = None,
    ) -> str:
        if self._script is None:
            raise AssertionError("FakeAnthropicClient: no script installed")
        return self._script.think_text

    def init_client(self) -> None:
        pass

    async def _messages_create(self, **kwargs: Any) -> Any:
        script = self._script
        if script is None:
            raise AssertionError("FakeAnthropicClient: no script installed")
        self.summary_create_calls.append(copy.deepcopy(kwargs))
        # Snapshot the active script before awaiting so in-flight compaction keeps using
        # the summary configuration that was installed for that request.
        if script.summary_delay:
            await asyncio.sleep(script.summary_delay)
        return SimpleNamespace(content=[SimpleNamespace(text=script.summary_text)])

    def reset(self) -> None:
        self._script = None
        self._round_cursor = 0
        self.summary_create_calls = []
        self.stream_context_partitions = []

    def set_script(self, script: LLMScript) -> None:
        self._script = script
        self._round_cursor = 0

    async def stream_turn(
        self,
        context_partition: ContextPartition,
        model: str,
        max_tool_call_rounds: int | None = None,
        on_usage: Callable[[int, int], None] | None = None,
        reasoning_effort: str | None = None,
        stream_ndjson: bool = False,
        tool_callbacks: dict[str, FunctionToolCallback] | None = None,
    ) -> AsyncGenerator[str, Any]:
        if self._script is None:
            raise AssertionError("FakeAnthropicClient: no script installed")
        self.stream_context_partitions.append(context_partition.model_copy(deep=True))

        answer_text: list[str] = []
        tool_call_rounds = 0
        while self._round_cursor < len(self._script.rounds):
            round_ = self._script.rounds[self._round_cursor]
            self._round_cursor += 1

            round_text = "".join(round_.text_deltas)
            if on_usage is not None:
                on_usage(round_.input_tokens, round_.output_tokens)
            context_window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
            context_pct = int(round_.input_tokens / context_window * 100)
            if stream_ndjson:
                yield json.dumps({"context_pct": context_pct}) + "\n"

            if round_.stop_reason != "tool_use":
                answer_text.extend(round_.text_deltas)
                if stream_ndjson:
                    for delta in round_.text_deltas:
                        yield json.dumps({"text_delta": delta}) + "\n"
                break

            if stream_ndjson and round_text:
                yield json.dumps({"progress_message": round_text}) + "\n"

            callback_tasks: list[asyncio.Task] = []
            for tc in round_.tool_calls:
                if stream_ndjson:
                    yield json.dumps({"tool_call": {"name": tc.name, "arguments": tc.arguments}}) + "\n"
                item = ContextPartitionItem(
                    arguments=tc.arguments,
                    call_id=tc.call_id,
                    id=tc.call_id,
                    name=tc.name,
                    status="completed",
                    type="function_call",
                )
                context_partition.append(item)
                if tool_callbacks and tc.name in tool_callbacks:
                    callback_tasks.append(
                        asyncio.create_task(tool_callbacks[tc.name].call(tc.arguments, tc.call_id))
                    )

            if not callback_tasks:
                break

            tool_call_rounds += 1
            if max_tool_call_rounds is not None and tool_call_rounds >= max_tool_call_rounds:
                break

            results = await asyncio.gather(*callback_tasks)
            if not all(r is not None for r in results):
                break
            context_partition.extend(r for r in results if r is not None)

        if answer_text:
            context_partition.append(
                ContextPartitionItem(content="".join(answer_text), role="assistant")
            )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self._round_cursor = 0
        self._script: LLMScript | None = None
        self.summary_create_calls: list[dict[str, Any]] = []
        self.stream_context_partitions: list[ContextPartition] = []
        self.async_openai = SimpleNamespace(
            responses=SimpleNamespace(create=self._responses_create),
        )

    async def close(self) -> None:
        pass

    async def complete(
        self,
        context_partition: ContextPartition,
        model: str,
        reasoning_effort: str | None = None,
    ) -> str:
        if self._script is None:
            raise AssertionError("FakeOpenAIClient: no script installed")
        return self._script.think_text

    def init_client(self) -> None:
        pass

    async def _responses_create(self, *, stream: bool = False, **kwargs: Any) -> Any:
        script = self._script
        if script is None:
            raise AssertionError("FakeOpenAIClient: no script installed")
        if stream:
            raise AssertionError(
                "FakeOpenAIClient.responses.create(stream=True) is not implemented; "
                "streaming goes through stream_turn()."
            )
        self.summary_create_calls.append(copy.deepcopy(kwargs))
        # Snapshot the active script before awaiting so in-flight compaction keeps using
        # the summary configuration that was installed for that request.
        if script.summary_delay:
            await asyncio.sleep(script.summary_delay)
        return SimpleNamespace(output_text=script.summary_text)

    def reset(self) -> None:
        self._script = None
        self._round_cursor = 0
        self.summary_create_calls = []
        self.stream_context_partitions = []

    def set_script(self, script: LLMScript) -> None:
        self._script = script
        self._round_cursor = 0

    async def stream_turn(
        self,
        context_partition: ContextPartition,
        model: str,
        max_tool_call_rounds: int | None = None,
        on_usage: Callable[[int, int], None] | None = None,
        reasoning_effort: str | None = None,
        stream_ndjson: bool = False,
        tool_callbacks: dict[str, FunctionToolCallback] | None = None,
    ) -> AsyncGenerator[str, Any]:
        if self._script is None:
            raise AssertionError("FakeOpenAIClient: no script installed")
        self.stream_context_partitions.append(context_partition.model_copy(deep=True))

        answer_text: list[str] = []
        tool_call_rounds = 0
        while self._round_cursor < len(self._script.rounds):
            round_ = self._script.rounds[self._round_cursor]
            self._round_cursor += 1

            round_text = "".join(round_.text_deltas)
            callback_tasks: list[asyncio.Task] = []

            if round_.stop_reason == "tool_use":
                for tc in round_.tool_calls:
                    if stream_ndjson:
                        yield json.dumps(
                            {"tool_call": {"name": tc.name, "arguments": tc.arguments or "{}"}}
                        ) + "\n"
                    item = ContextPartitionItem(
                        arguments=tc.arguments,
                        call_id=tc.call_id,
                        id=tc.call_id,
                        name=tc.name,
                        status="completed",
                        type="function_call",
                    )
                    context_partition.append(item)
                    if tool_callbacks and tc.name in tool_callbacks:
                        callback_tasks.append(
                            asyncio.create_task(tool_callbacks[tc.name].call(tc.arguments, tc.call_id))
                        )

            if on_usage is not None:
                on_usage(round_.input_tokens, round_.output_tokens)
            context_window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
            context_pct = int(round_.input_tokens / context_window * 100)
            if stream_ndjson:
                yield json.dumps({"context_pct": context_pct}) + "\n"

            if round_.stop_reason == "tool_use":
                if stream_ndjson and round_text:
                    yield json.dumps({"progress_message": round_text}) + "\n"
                if not callback_tasks:
                    break
                tool_call_rounds += 1
                if max_tool_call_rounds is not None and tool_call_rounds >= max_tool_call_rounds:
                    break
                results = await asyncio.gather(*callback_tasks)
                if not all(r is not None for r in results):
                    break
                context_partition.extend(r for r in results if r is not None)
                continue

            answer_text.extend(round_.text_deltas)
            if stream_ndjson:
                for delta in round_.text_deltas:
                    yield json.dumps({"text_delta": delta}) + "\n"
            break

        if answer_text:
            context_partition.append(
                ContextPartitionItem(content="".join(answer_text), role="assistant")
            )
