"""OpenAIClient — pre-projected `list[ProjectedItem]` in, streaming events out.

Owns its own `ProjectedItem → OpenAI Responses input` translation and its own working buffer for the in-flight
turn. Intermediate assistant narration is transient (not emitted via `on_committed_turn_item`). The final
assistant text is delivered via `on_final_assistant_message`; committed tool items via `on_committed_turn_item`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator, Callable
from typing import Any

from openai import AsyncOpenAI
from openai.types.responses import ResponseStreamEvent, ToolParam
from openai.types.shared_params import Reasoning

from prokaryotes.api_v1.models import FunctionToolCallback, LLMClient
from prokaryotes.conversation_v1.models import ProjectedItem, TurnItem
from prokaryotes.utils_v1.llm_utils import DEFAULT_CONTEXT_WINDOW, MODEL_CONTEXT_WINDOWS

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    def __init__(self):
        self.async_openai: AsyncOpenAI | None = None

    async def close(self):
        await self.async_openai.close()

    async def complete(
        self,
        items: list[ProjectedItem],
        instruction: str | None,
        model: str,
        reasoning_effort: str | None = None,
    ) -> str:
        input_items = _items_to_openai_input(items, instruction)
        reasoning = Reasoning(effort=reasoning_effort if reasoning_effort else "none")
        response = await self.async_openai.responses.create(  # type: ignore
            model=model,
            input=input_items,
            reasoning=reasoning,
            stream=False,
        )
        return "".join(
            block.text
            for item in response.output
            if item.type == "message"
            for block in item.content
            if block.type == "output_text"
        )

    def init_client(self):
        self.async_openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        answer_text: list[str] = []
        working_input = _items_to_openai_input(items, instruction)
        reasoning = Reasoning(effort=reasoning_effort if reasoning_effort else "none")
        tool_call_rounds = 0
        tool_params: list[ToolParam] | None = (
            [cb.tool_spec.to_openai_function_tool_param() for cb in tool_callbacks.values()] if tool_callbacks else None
        )
        if working_input and working_input[0].get("role") == "developer":
            logger.info("LLM developer message:\n%s", working_input[0].get("content", ""))

        while True:
            round_text: list[str] = []
            tool_call_seen = [False]
            # Pair each function_call with its in-flight task (or None when no callback is registered). Defer
            # `on_committed_turn_item` until the round provably finishes — committing a function_call without its
            # output would leave an orphan in the persisted `TurnExecution`.
            pending_calls: list[tuple[TurnItem, asyncio.Task[TurnItem | None] | None]] = []

            async for event in await self.async_openai.responses.create(  # type: ignore
                model=model,
                input=working_input,
                tools=tool_params,
                reasoning=reasoning,
                stream=True,
            ):
                str_to_yield = await self._handle_event(
                    event=event,
                    on_usage=on_usage,
                    model=model,
                    ndjson=stream_ndjson,
                    pending_calls=pending_calls,
                    round_text=round_text,
                    tool_call_seen=tool_call_seen,
                    tool_callbacks=tool_callbacks,
                    working_input=working_input,
                )
                if str_to_yield:
                    yield str_to_yield

            round_output = "".join(round_text)
            if tool_call_seen[0]:
                if stream_ndjson and round_output:
                    yield json.dumps({"progress_message": round_output}) + "\n"
                callback_tasks = [t for _, t in pending_calls if t is not None]
                if not callback_tasks:
                    break
                # Await dispatched tasks before the round-limit check so the max-rounds branch never abandons
                # in-flight callbacks.
                results = await asyncio.gather(*callback_tasks)
                if not all(r is not None for r in results):
                    break
                # Atomically commit fc_item + its output for each completed pair. Pairs with `task is None`
                # (unregistered tool) are dropped so no orphan function_call lands in the persisted
                # `TurnExecution`. The commit happens *before* the round-limit check so the work this round
                # produced is always reflected in storage — the limit only blocks the next LLM call.
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
                    working_input.append(_turn_item_to_openai_dict(result))
                tool_call_rounds += 1
                if max_tool_call_rounds is not None and tool_call_rounds >= max_tool_call_rounds:
                    logger.warning("Reached max_tool_call_rounds=%d; stopping", max_tool_call_rounds)
                    break
                if round_output and not stream_ndjson:
                    yield "\n"
                continue

            answer_text.extend(round_text)
            if stream_ndjson:
                for delta in round_text:
                    yield json.dumps({"text_delta": delta}) + "\n"
            break

        if answer_text and on_final_assistant_message is not None:
            on_final_assistant_message("".join(answer_text))

    async def _handle_event(
        self,
        *,
        event: ResponseStreamEvent,
        on_usage: Callable[[int, int], None] | None,
        model: str,
        ndjson: bool,
        pending_calls: list[tuple[TurnItem, asyncio.Task[TurnItem | None] | None]],
        round_text: list[str],
        tool_call_seen: list[bool],
        tool_callbacks: dict[str, FunctionToolCallback] | None,
        working_input: list[dict],
    ) -> str | None:
        if event.type == "response.output_text.delta":
            round_text.append(event.delta)
            if not ndjson:
                return event.delta
        elif event.type == "response.output_text.done":
            pass
        elif event.type == "response.output_item.done" and event.item.type == "function_call":
            logger.info("Invoking callback %s with arguments %s", event.item.name, event.item.arguments)
            tool_call_seen[0] = True
            fc_item = TurnItem(
                arguments=event.item.arguments,
                call_id=event.item.call_id,
                id=getattr(event.item, "id", None),
                name=event.item.name,
                type="function_call",
                status=getattr(event.item, "status", "completed"),
            )
            # Persistence is deferred to the main loop's atomic commit (after gather + round-limit check), so
            # mirror into `working_input` to keep later rounds in this stream consistent.
            working_input.append(_turn_item_to_openai_dict(fc_item))
            task: asyncio.Task[TurnItem | None] | None = None
            if tool_callbacks and event.item.name in tool_callbacks:
                task = asyncio.create_task(
                    tool_callbacks[event.item.name].call(event.item.arguments, event.item.call_id)
                )
            else:
                logger.warning("No callback for tool %r", event.item.name)
            pending_calls.append((fc_item, task))
            if ndjson:
                return json.dumps({"tool_call": {"name": fc_item.name, "arguments": fc_item.arguments or "{}"}}) + "\n"
        elif event.type == "response.completed":
            if on_usage is not None:
                usage = event.response.usage
                on_usage(usage.input_tokens, usage.output_tokens)
            context_win = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
            context_pct = int(event.response.usage.input_tokens / context_win * 100)
            return (json.dumps({"context_pct": context_pct}) + "\n") if ndjson else None
        return None


def _items_to_openai_input(items: list[ProjectedItem], instruction: str | None) -> list[dict]:
    """ProjectedItem → OpenAI Responses input list. Instruction is injected as a leading `developer` message
    (OpenAI's equivalent of `system`)."""
    result: list[dict] = []
    if instruction:
        result.append({"role": "developer", "content": instruction, "type": "message"})
    for item in items:
        if item.type == "message":
            if item.role is None:
                continue
            role = "developer" if item.role == "system" else item.role
            result.append({"role": role, "content": item.content or "", "type": "message"})
        elif item.type == "function_call":
            entry: dict[str, Any] = {"type": "function_call"}
            if item.call_id:
                entry["call_id"] = item.call_id
            if item.name:
                entry["name"] = item.name
            if item.arguments is not None:
                entry["arguments"] = item.arguments
            result.append(entry)
        elif item.type == "function_call_output":
            entry = {"type": "function_call_output"}
            if item.call_id:
                entry["call_id"] = item.call_id
            if item.output is not None:
                entry["output"] = item.output
            result.append(entry)
    return result


def _turn_item_to_openai_dict(item: TurnItem) -> dict:
    entry: dict[str, Any] = {"type": item.type}
    if item.call_id:
        entry["call_id"] = item.call_id
    if item.type == "function_call":
        if item.name:
            entry["name"] = item.name
        if item.arguments is not None:
            entry["arguments"] = item.arguments
        if item.status is not None:
            entry["status"] = item.status
    elif item.type == "function_call_output":
        if item.output is not None:
            entry["output"] = item.output
    return entry
