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
        max_tool_call_rounds: int = None,
        on_usage: Callable[[int, int], None] | None = None,
        reasoning_effort: str = None,
        stream_ndjson: bool = False,
        tool_callbacks: dict[str, FunctionToolCallback] = None,
        tool_choice: dict[str, Any] = None,
    ) -> AsyncGenerator[str, Any]:
        tool_params = (
            [cb.tool_spec.to_anthropic_tool_param() for cb in tool_callbacks.values()]
            if tool_callbacks else None
        )
        thinking_budget = {"low": 1024, "medium": 2048, "high": 4096}.get(reasoning_effort or "")
        thinking = {"type": "enabled", "budget_tokens": thinking_budget} if thinking_budget else None
        tool_call_rounds = 0
        text_yielded = False

        while True:
            if tool_call_rounds > 0 and text_yielded:
                yield (json.dumps({"text_delta": "\n"}) + "\n") if stream_ndjson else "\n"
            text_yielded = False
            system, messages = context_partition.to_anthropic_messages()
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
                    yield (json.dumps({"text_delta": delta}) + "\n") if stream_ndjson else delta
                response = await stream.get_final_message()

            if on_usage is not None:
                on_usage(response.usage.input_tokens, response.usage.output_tokens)

            text_content = "".join(b.text for b in response.content if b.type == "text")
            if text_content:
                context_partition.append(ContextPartitionItem(role="assistant", content=text_content))

            if response.stop_reason != "tool_use":
                break

            callback_tasks: list[asyncio.Task] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                args = json.dumps(block.input, separators=(",", ":"))
                item = ContextPartitionItem(
                    id=block.id, call_id=block.id, name=block.name,
                    arguments=args, type="function_call", status="completed",
                )
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
