"""AnthropicClient — pre-projected `list[ProjectedItem]` in, streaming events out.

Owns its own `ProjectedItem → Anthropic message format` translation and its own working buffer for the in-flight
turn. Intermediate assistant narration goes into the working buffer so subsequent rounds in the tool-use loop see it,
but is *not* committed to the `TurnExecution` (transient-narration invariant). The final assistant text is delivered
via `on_final_assistant_message`; committed tool items via `on_committed_turn_item`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator, Callable
from typing import Any

from anthropic import AsyncAnthropic

from prokaryotes.api_v1.models import FunctionToolCallback, LLMClient
from prokaryotes.conversation_v1.models import ProjectedItem, TurnItem
from prokaryotes.utils_v1.llm_utils import DEFAULT_CONTEXT_WINDOW, MODEL_CONTEXT_WINDOWS

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    def __init__(self):
        self.async_anthropic: AsyncAnthropic | None = None

    async def close(self):
        if self.async_anthropic is not None:
            await self.async_anthropic.close()

    async def complete(
        self,
        items: list[ProjectedItem],
        instruction: str | None,
        model: str,
        reasoning_effort: str | None = None,
    ) -> str:
        thinking_budget = {"low": 1024, "medium": 2048, "high": 4096}.get(reasoning_effort or "")
        thinking = {"type": "enabled", "budget_tokens": thinking_budget} if thinking_budget else None
        messages = _items_to_anthropic_messages(items)
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
            "messages": messages,
        }
        if instruction:
            params["system"] = instruction
        if thinking:
            params["thinking"] = thinking
        response = await self.async_anthropic.messages.create(**params)
        return "".join(block.text for block in response.content if block.type == "text")

    def init_client(self):
        self.async_anthropic = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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
        """Working-buffer state machine over Anthropic messages.

        Per-round translation: starts from the initial `items` and grows as tool_use/tool_result blocks are
        produced. Intermediate text narration flows into the buffer so the model sees its own scratch work between
        rounds, then is discarded at turn finalization.
        """
        tool_params = (
            [cb.tool_spec.to_anthropic_tool_param() for cb in tool_callbacks.values()] if tool_callbacks else None
        )
        thinking_budget = {"low": 1024, "medium": 2048, "high": 4096}.get(reasoning_effort or "")
        thinking = {"type": "enabled", "budget_tokens": thinking_budget} if thinking_budget else None
        tool_call_rounds = 0

        # Working buffer: initialized from the projection, grows with tool_use/result blocks, discarded when
        # stream_turn returns.
        working_messages = _items_to_anthropic_messages(items)
        # Text aggregated across all non-tool-use rounds.
        answer_text: list[str] = []

        while True:
            if tool_call_rounds == 0 and instruction:
                logger.info("LLM system message:\n%s", instruction)
            params: dict[str, Any] = {
                "model": model,
                "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
                "messages": working_messages,
            }
            if instruction:
                params["system"] = instruction
            if tool_params:
                params["tools"] = tool_params
            if thinking:
                params["thinking"] = thinking

            round_text: list[str] = []
            async with self.async_anthropic.messages.stream(**params) as stream:
                async for delta in stream.text_stream:
                    round_text.append(delta)
                    if not stream_ndjson:
                        yield delta
                response = await stream.get_final_message()

            total_input = (
                response.usage.input_tokens
                + getattr(response.usage, "cache_read_input_tokens", 0)
                + getattr(response.usage, "cache_creation_input_tokens", 0)
            )
            if on_usage is not None:
                on_usage(total_input, response.usage.output_tokens)
            context_window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
            context_pct = int(total_input / context_window * 100)
            if stream_ndjson:
                yield json.dumps({"context_pct": context_pct}) + "\n"

            if response.stop_reason != "tool_use":
                # Final assistant text. Emit via callback for commit to Conversation.
                answer_text.extend(round_text)
                if stream_ndjson:
                    for delta in round_text:
                        yield json.dumps({"text_delta": delta}) + "\n"
                break

            # Intermediate text is transient — feed back to the model in subsequent rounds but do NOT emit via
            # on_committed_turn_item.
            round_output = "".join(round_text)
            if stream_ndjson and round_output:
                yield json.dumps({"progress_message": round_output}) + "\n"

            # Append the model's assistant turn (text + tool_use blocks) to the buffer.
            assistant_blocks: list[dict] = []
            if round_output:
                assistant_blocks.append({"type": "text", "text": round_output})
            # Pair each fc_item with its in-flight task (or None when no callback is registered). Defer
            # `on_committed_turn_item` until the round provably finishes: a fc_item committed without its
            # function_call_output would be an orphan in the persisted `TurnExecution` and a tool_use without a
            # tool_result on the next API call.
            pending_calls: list[tuple[TurnItem, asyncio.Task[TurnItem | None] | None]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                args = json.dumps(block.input, separators=(",", ":"))
                if stream_ndjson:
                    yield json.dumps({"tool_call": {"name": block.name, "arguments": args}}) + "\n"
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
                fc_item = TurnItem(
                    id=block.id,
                    call_id=block.id,
                    name=block.name,
                    arguments=args,
                    type="function_call",
                    status="completed",
                )
                task: asyncio.Task[TurnItem | None] | None = None
                if tool_callbacks and block.name in tool_callbacks:
                    task = asyncio.create_task(tool_callbacks[block.name].call(args, block.id))
                else:
                    logger.warning("No callback for tool %r", block.name)
                pending_calls.append((fc_item, task))

            if assistant_blocks:
                working_messages.append({"role": "assistant", "content": assistant_blocks})

            callback_tasks = [t for _, t in pending_calls if t is not None]
            if not callback_tasks:
                break

            # Await all dispatched tasks before checking the round-limit so the max-rounds branch never abandons
            # in-flight callbacks.
            results = await asyncio.gather(*callback_tasks)
            if not all(r is not None for r in results):
                break

            # Atomically commit fc_item + its tool_result for each successfully completed pair. Pairs whose
            # `task is None` (unregistered tool) are dropped entirely so neither the fc_item nor a synthetic output
            # lands in the persisted `TurnExecution`. The commit happens *before* the round-limit check so the work
            # this round actually produced is always reflected in storage — the limit only blocks the next LLM call.
            result_iter = iter(results)
            tool_result_blocks: list[dict] = []
            for fc_item, task in pending_calls:
                if task is None:
                    continue
                result = next(result_iter)
                if result is None:
                    continue
                if on_committed_turn_item is not None:
                    on_committed_turn_item(fc_item)
                    on_committed_turn_item(result)
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": fc_item.id,
                        "content": result.output or "",
                    }
                )
            if tool_result_blocks:
                working_messages.append({"role": "user", "content": tool_result_blocks})

            tool_call_rounds += 1
            if max_tool_call_rounds is not None and tool_call_rounds >= max_tool_call_rounds:
                logger.warning("Reached max_tool_call_rounds=%d; stopping", max_tool_call_rounds)
                break
            if round_output and not stream_ndjson:
                yield "\n"

        if answer_text and on_final_assistant_message is not None:
            on_final_assistant_message("".join(answer_text))


def _items_to_anthropic_messages(items: list[ProjectedItem]) -> list[dict]:
    """Group `ProjectedItem`s into Anthropic `{role, content[]}` messages.

    `type=message` items become text blocks; `function_call`/`function_call_output` become `tool_use` /
    `tool_result` blocks. Consecutive items of the same role coalesce into a single message — already enforced by
    `project_for_llm`'s same-role merge, but we re-group here so the wire shape is invariant under ad-hoc callers.
    """
    messages: list[dict] = []
    current_role: str | None = None
    current_content: list[dict] = []

    def flush():
        nonlocal current_role, current_content
        if current_role and current_content:
            messages.append({"role": current_role, "content": current_content})
        current_role, current_content = None, []

    for item in items:
        if item.type == "message":
            if item.role == "system":
                # Should not reach the LLM client — instruction is the `system` param. Drop defensively.
                flush()
                continue
            if item.role not in {"user", "assistant"}:
                raise ValueError(f"Unsupported role: {item.role!r}")
            if not item.content:
                # Anthropic rejects empty text blocks; a content-less message contributes nothing — skip it rather
                # than emit `{"text": ""}`.
                continue
            role, block = item.role, {"type": "text", "text": item.content}
        elif item.type == "function_call":
            call_id = item.call_id
            if call_id is None or item.name is None:
                raise ValueError("function_call requires call_id and name")
            role, block = (
                "assistant",
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": item.name,
                    "input": json.loads(item.arguments or "{}"),
                },
            )
        elif item.type == "function_call_output":
            call_id = item.call_id
            if call_id is None:
                raise ValueError("function_call_output requires call_id")
            role, block = (
                "user",
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": item.output or "",
                },
            )
        else:
            raise ValueError(f"Unsupported ProjectedItem type: {item.type!r}")

        if current_role != role:
            flush()
            current_role = role
        current_content.append(block)

    flush()
    return messages
