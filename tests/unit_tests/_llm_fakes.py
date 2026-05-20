"""Scriptable fake LLM clients matching the unified-conversation contract.

Substitutes at the harness-contract level: `stream_turn` accepts the new `(items: list[ProjectedItem],
instruction: str | None, ...)` signature with `on_committed_turn_item` / `on_final_assistant_message` callbacks.
Intermediate narration in tool-use rounds is emitted as `progress_message` NDJSON events but NEVER passed to
`on_committed_turn_item` (transient-narration invariant).

Recording hooks (`stream_turn_calls`, `complete_calls`) let tests assert on exactly what reached the LLM layer.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any

from prokaryotes.api_v1.models import FunctionToolCallback
from prokaryotes.conversation_v1.models import ProjectedItem, TurnItem
from prokaryotes.utils_v1.llm_utils import DEFAULT_CONTEXT_WINDOW, MODEL_CONTEXT_WINDOWS


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
    # Anthropic-only: count toward `cached_input_tokens` in the on_usage callback.
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class LLMScript:
    rounds: list[LLMRound]
    summary_delay: float = 0.0
    summary_text: str = "STUB SUMMARY"
    think_text: str = "STUB THINK ANALYSIS"


class _FakeLLMClientBase:
    """Common scripting state and recording hooks for both fakes."""

    def __init__(self) -> None:
        self._round_cursor = 0
        self._script: LLMScript | None = None
        # Recording hooks.
        self.complete_calls: list[dict[str, Any]] = []
        self.stream_turn_calls: list[dict[str, Any]] = []

    async def close(self) -> None:
        pass

    def init_client(self) -> None:
        pass

    def reset(self) -> None:
        self._script = None
        self._round_cursor = 0
        self.complete_calls = []
        self.stream_turn_calls = []

    def set_script(self, script: LLMScript) -> None:
        self._script = script
        self._round_cursor = 0

    async def complete(
        self,
        items: list[ProjectedItem],
        instruction: str | None,
        model: str,
        reasoning_effort: str | None = None,
    ) -> str:
        if self._script is None:
            raise AssertionError(f"{type(self).__name__}: no script installed")
        self.complete_calls.append(
            {
                "items": [item.model_copy(deep=True) for item in items],
                "instruction": instruction,
                "model": model,
                "reasoning_effort": reasoning_effort,
            }
        )
        if self._script.summary_delay:
            await asyncio.sleep(self._script.summary_delay)
        # Summarization (`WebHarness._summarize_and_compact`) passes `instruction=None`; the think tool passes a
        # non-None `instruction` (its system prompt).
        if instruction is None:
            return self._script.summary_text
        return self._script.think_text

    def _record_stream_turn_call(
        self,
        *,
        items: list[ProjectedItem],
        instruction: str | None,
        model: str,
        reasoning_effort: str | None,
        max_tool_call_rounds: int | None,
        stream_ndjson: bool,
        tool_callback_names: list[str],
    ) -> None:
        self.stream_turn_calls.append(
            {
                "items": [item.model_copy(deep=True) for item in items],
                "instruction": instruction,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "max_tool_call_rounds": max_tool_call_rounds,
                "stream_ndjson": stream_ndjson,
                "tool_callback_names": list(tool_callback_names),
            }
        )


class FakeAnthropicClient(_FakeLLMClientBase):
    """Fake AnthropicClient matching the overlay's stream_turn / complete contract."""

    async def stream_turn(
        self,
        items: list[ProjectedItem],
        instruction: str | None,
        model: str,
        max_tool_call_rounds: int | None = None,
        on_committed_turn_item: Callable[[TurnItem], None] | None = None,
        on_final_assistant_message: Callable[[str], None] | None = None,
        on_usage: Callable[[int, int], None] | None = None,
        reasoning_effort: str | None = None,
        stream_ndjson: bool = False,
        tool_callbacks: dict[str, FunctionToolCallback] | None = None,
    ) -> AsyncGenerator[str, Any]:
        if self._script is None:
            raise AssertionError("FakeAnthropicClient: no script installed")
        self._record_stream_turn_call(
            items=items,
            instruction=instruction,
            model=model,
            reasoning_effort=reasoning_effort,
            max_tool_call_rounds=max_tool_call_rounds,
            stream_ndjson=stream_ndjson,
            tool_callback_names=list(tool_callbacks.keys()) if tool_callbacks else [],
        )

        answer_text: list[str] = []
        tool_call_rounds = 0
        while self._round_cursor < len(self._script.rounds):
            round_ = self._script.rounds[self._round_cursor]
            self._round_cursor += 1

            round_text = "".join(round_.text_deltas)
            total_input = round_.input_tokens + round_.cache_read_input_tokens + round_.cache_creation_input_tokens
            if on_usage is not None:
                on_usage(total_input, round_.output_tokens)
            context_window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
            context_pct = int(total_input / context_window * 100)
            if stream_ndjson:
                yield json.dumps({"context_pct": context_pct}) + "\n"

            if round_.stop_reason != "tool_use":
                # Final-text round. Emit deltas and break.
                answer_text.extend(round_.text_deltas)
                if stream_ndjson:
                    for delta in round_.text_deltas:
                        yield json.dumps({"text_delta": delta}) + "\n"
                else:
                    for delta in round_.text_deltas:
                        yield delta
                break

            # Tool-use round: intermediate narration is transient. Emit as progress_message NDJSON, but NEVER
            # call on_committed_turn_item with it.
            if stream_ndjson and round_text:
                yield json.dumps({"progress_message": round_text}) + "\n"

            # Pair fc_item with its in-flight task; defer commits to after gather + round-limit check so the
            # fakes mirror the production contract against orphan function_call persistence.
            pending_calls: list[tuple[TurnItem, asyncio.Task[TurnItem | None] | None]] = []
            for tc in round_.tool_calls:
                if stream_ndjson:
                    yield json.dumps({"tool_call": {"name": tc.name, "arguments": tc.arguments}}) + "\n"
                fc_item = TurnItem(
                    arguments=tc.arguments,
                    call_id=tc.call_id,
                    id=tc.call_id,
                    name=tc.name,
                    status="completed",
                    type="function_call",
                )
                task: asyncio.Task[TurnItem | None] | None = None
                if tool_callbacks and tc.name in tool_callbacks:
                    task = asyncio.create_task(tool_callbacks[tc.name].call(tc.arguments, tc.call_id))
                pending_calls.append((fc_item, task))

            callback_tasks = [t for _, t in pending_calls if t is not None]
            if not callback_tasks:
                break

            results = await asyncio.gather(*callback_tasks)
            if not all(r is not None for r in results):
                break

            # Commit this round's work, then check the limit. The limit only blocks the next LLM call.
            result_iter = iter(results)
            for fc_item, task in pending_calls:
                if task is None:
                    continue
                result = next(result_iter)
                if result is None:
                    continue
                if on_committed_turn_item is not None:
                    on_committed_turn_item(fc_item)
                    on_committed_turn_item(result)

            tool_call_rounds += 1
            if max_tool_call_rounds is not None and tool_call_rounds >= max_tool_call_rounds:
                break

        if answer_text and on_final_assistant_message is not None:
            on_final_assistant_message("".join(answer_text))


class FakeOpenAIClient(_FakeLLMClientBase):
    """Fake OpenAIClient matching the overlay's stream_turn / complete contract.

    Mirrors FakeAnthropicClient but emits events in OpenAI's order (tool_call before context_pct within a
    tool-use round). The on_committed_turn_item / on_final_assistant_message contract is identical.
    """

    async def stream_turn(
        self,
        items: list[ProjectedItem],
        instruction: str | None,
        model: str,
        max_tool_call_rounds: int | None = None,
        on_committed_turn_item: Callable[[TurnItem], None] | None = None,
        on_final_assistant_message: Callable[[str], None] | None = None,
        on_usage: Callable[[int, int], None] | None = None,
        reasoning_effort: str | None = None,
        stream_ndjson: bool = False,
        tool_callbacks: dict[str, FunctionToolCallback] | None = None,
    ) -> AsyncGenerator[str, Any]:
        if self._script is None:
            raise AssertionError("FakeOpenAIClient: no script installed")
        self._record_stream_turn_call(
            items=items,
            instruction=instruction,
            model=model,
            reasoning_effort=reasoning_effort,
            max_tool_call_rounds=max_tool_call_rounds,
            stream_ndjson=stream_ndjson,
            tool_callback_names=list(tool_callbacks.keys()) if tool_callbacks else [],
        )

        answer_text: list[str] = []
        tool_call_rounds = 0
        while self._round_cursor < len(self._script.rounds):
            round_ = self._script.rounds[self._round_cursor]
            self._round_cursor += 1

            round_text = "".join(round_.text_deltas)
            # Pair fc_item with its in-flight task; defer commits to after gather + round-limit check so the
            # fakes mirror the production contract against orphan function_call persistence.
            pending_calls: list[tuple[TurnItem, asyncio.Task[TurnItem | None] | None]] = []

            if round_.stop_reason == "tool_use":
                for tc in round_.tool_calls:
                    if stream_ndjson:
                        yield json.dumps({"tool_call": {"name": tc.name, "arguments": tc.arguments or "{}"}}) + "\n"
                    fc_item = TurnItem(
                        arguments=tc.arguments,
                        call_id=tc.call_id,
                        id=tc.call_id,
                        name=tc.name,
                        status="completed",
                        type="function_call",
                    )
                    task: asyncio.Task[TurnItem | None] | None = None
                    if tool_callbacks and tc.name in tool_callbacks:
                        task = asyncio.create_task(tool_callbacks[tc.name].call(tc.arguments, tc.call_id))
                    pending_calls.append((fc_item, task))

            if on_usage is not None:
                on_usage(round_.input_tokens, round_.output_tokens)
            context_window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
            context_pct = int(round_.input_tokens / context_window * 100)
            if stream_ndjson:
                yield json.dumps({"context_pct": context_pct}) + "\n"

            if round_.stop_reason == "tool_use":
                if stream_ndjson and round_text:
                    yield json.dumps({"progress_message": round_text}) + "\n"
                callback_tasks = [t for _, t in pending_calls if t is not None]
                if not callback_tasks:
                    break
                results = await asyncio.gather(*callback_tasks)
                if not all(r is not None for r in results):
                    break
                # Commit this round's work, then check the limit. The limit only blocks the next LLM call.
                result_iter = iter(results)
                for fc_item, task in pending_calls:
                    if task is None:
                        continue
                    result = next(result_iter)
                    if result is None:
                        continue
                    if on_committed_turn_item is not None:
                        on_committed_turn_item(fc_item)
                        on_committed_turn_item(result)
                tool_call_rounds += 1
                if max_tool_call_rounds is not None and tool_call_rounds >= max_tool_call_rounds:
                    break
                continue

            answer_text.extend(round_.text_deltas)
            if stream_ndjson:
                for delta in round_.text_deltas:
                    yield json.dumps({"text_delta": delta}) + "\n"
            else:
                for delta in round_.text_deltas:
                    yield delta
            break

        if answer_text and on_final_assistant_message is not None:
            on_final_assistant_message("".join(answer_text))


# Convenience builder for the common one-round all-text script.
def text_only_script(text: str, *, input_tokens: int = 1000, output_tokens: int = 200) -> LLMScript:
    return LLMScript(
        rounds=[
            LLMRound(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reason="end_turn",
                text_deltas=[text],
            )
        ]
    )


# Convenience builder for one tool-use round followed by one final text round.
def tool_then_text_script(
    *,
    tool_name: str,
    tool_arguments: str,
    tool_call_id: str = "call-1",
    final_text: str = "Done.",
    intermediate_narration: str = "",
    input_tokens: int = 1000,
    output_tokens: int = 200,
) -> LLMScript:
    return LLMScript(
        rounds=[
            LLMRound(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reason="tool_use",
                text_deltas=[intermediate_narration] if intermediate_narration else [],
                tool_calls=[ToolCallSpec(arguments=tool_arguments, call_id=tool_call_id, name=tool_name)],
            ),
            LLMRound(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reason="end_turn",
                text_deltas=[final_text],
            ),
        ]
    )
