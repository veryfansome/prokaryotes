import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator, Callable
from typing import Any

from anthropic import AsyncAnthropic

from prokaryotes.api_v1.models import (
    ContextPartition,
    ContextPartitionItem,
    FunctionToolCallback,
)
from prokaryotes.utils_v1.llm_utils import (
    DEFAULT_CONTEXT_WINDOW,
    MODEL_CONTEXT_WINDOWS,
)

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.async_anthropic: AsyncAnthropic | None = None

    def init_client(self):
        self.async_anthropic = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def close(self):
        if self.async_anthropic is not None:
            await self.async_anthropic.close()

    async def stream_response(
        self,
        context_partition: ContextPartition,
        model: str,
        max_tool_call_rounds: int | None = None,
        on_usage: Callable[[int, int], None] | None = None,
        reasoning_effort: str | None = None,
        stream_ndjson: bool = False,
        tool_callbacks: dict[str, FunctionToolCallback] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, Any]:
        tool_params = (
            [cb.tool_spec.to_anthropic_tool_param() for cb in tool_callbacks.values()]
            if tool_callbacks else None
        )
        thinking_budget = {"low": 1024, "medium": 2048, "high": 4096}.get(reasoning_effort or "")
        thinking = {"type": "enabled", "budget_tokens": thinking_budget} if thinking_budget else None
        tool_call_rounds = 0
        text_yielded = False
        all_yielded_text: list[str] = []

        while True:
            if tool_call_rounds > 0 and text_yielded:
                sep = "\n"
                all_yielded_text.append(sep)
                yield (json.dumps({"text_delta": sep}) + "\n") if stream_ndjson else sep
            text_yielded = False
            round_text_start = len(all_yielded_text)
            system, messages = context_partition.to_anthropic_messages()
            if tool_call_rounds == 0 and system:
                logger.info(f"LLM system message:\n{system}")
            params: dict[str, Any] = {
                "model": model,
                "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
                "messages": messages,
            }
            if system:
                params["system"] = system
            if tool_params:
                params["tools"] = tool_params
                params["tool_choice"] = tool_choice or {"type": "auto"}
            if thinking:
                params["thinking"] = thinking

            async with self.async_anthropic.messages.stream(**params) as stream:
                async for delta in stream.text_stream:
                    text_yielded = True
                    all_yielded_text.append(delta)
                    yield (json.dumps({"text_delta": delta}) + "\n") if stream_ndjson else delta
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
                break

            round_text = "".join(all_yielded_text[round_text_start:])
            callback_tasks: list[asyncio.Task] = []
            first_tool_call = True
            for block in response.content:
                if block.type != "tool_use":
                    continue
                args = json.dumps(block.input, separators=(",", ":"))
                item = ContextPartitionItem(
                    id=block.id, call_id=block.id, name=block.name,
                    arguments=args, type="function_call", status="completed",
                    text_preamble=round_text if first_tool_call and round_text else None,
                )
                first_tool_call = False
                logger.info("Invoking callback %s with arguments %s", item.name, item.arguments)
                context_partition.append(item)
                if tool_callbacks and block.name in tool_callbacks:
                    callback_tasks.append(
                        asyncio.create_task(tool_callbacks[block.name].call(args, block.id))
                    )
                else:
                    logger.warning("No callback for tool %r", block.name)

            if not callback_tasks:
                break

            tool_call_rounds += 1
            if max_tool_call_rounds is not None and tool_call_rounds >= max_tool_call_rounds:
                logger.warning("Reached max_tool_call_rounds=%d, stopping", max_tool_call_rounds)
                break

            results = await asyncio.gather(*callback_tasks)
            if not all(r is not None for r in results):
                break
            context_partition.extend(r for r in results if r is not None)

        if all_yielded_text:
            context_partition.append(ContextPartitionItem(
                role="assistant", content="".join(all_yielded_text)
            ))
