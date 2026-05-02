import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator, Callable
from typing import Any

from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseStreamEvent,
    ToolParam,
)
from openai.types.shared_params import Reasoning

from prokaryotes.api_v1.models import (
    ContextPartition,
    ContextPartitionItem,
    FunctionToolCallback,
    LLMClient,
)
from prokaryotes.utils_v1.llm_utils import (
    DEFAULT_CONTEXT_WINDOW,
    MODEL_CONTEXT_WINDOWS,
)

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    def __init__(self):
        self.async_openai: AsyncOpenAI | None = None

    async def close(self):
        await self.async_openai.close()

    async def complete(
            self,
            context_partition: ContextPartition,
            model: str,
            reasoning_effort: str | None = None,
    ) -> str:
        input_items = context_partition.to_openai_input()
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

    @staticmethod
    async def handle_response_stream_event(
            callback_tasks: list[asyncio.Task[ContextPartitionItem | None]],
            context_window: ContextPartition,
            event: ResponseStreamEvent,
            tool_callbacks: dict[str, FunctionToolCallback],
            accumulated_text: list[str] | None = None,
            model: str = "",
            ndjson: bool = False,
            on_usage: Callable[[int, int], None] | None = None,
            round_text: list[str] | None = None,
    ) -> str | None:
        if event.type == "response.output_text.delta":
            if accumulated_text is not None:
                accumulated_text.append(event.delta)
            if round_text is not None:
                round_text.append(event.delta)
            return (json.dumps({"text_delta": event.delta}) + "\n") if ndjson else event.delta
        elif event.type == "response.output_text.done":
            pass  # combined assistant item appended after all rounds complete
        elif event.type == "response.output_item.done" and event.item.type == "function_call":
            logger.info(f"Invoking callback {event.item.name} with arguments {event.item.arguments}")
            preamble = "".join(round_text) if round_text else None
            if round_text is not None:
                round_text.clear()
            item = ContextPartitionItem(**event.item.__dict__)
            if preamble:
                item.text_preamble = preamble
            context_window.append(item)
            callback_task: asyncio.Task[ContextPartitionItem | None] = asyncio.create_task(
                tool_callbacks[event.item.name].call(
                    event.item.arguments,
                    event.item.call_id,
                )
            )
            callback_tasks.append(callback_task)
        elif event.type == "response.completed":
            if on_usage is not None:
                usage = event.response.usage
                on_usage(usage.input_tokens, usage.output_tokens)
            context_win = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
            context_pct = int(event.response.usage.input_tokens / context_win * 100)
            return (json.dumps({"context_pct": context_pct}) + "\n") if ndjson else None
        elif (event.type.startswith("response.web_search_call")
              or (event.type == "response.output_item.done" and event.item.type == "search")):
            logger.info(event)
        return None

    def init_client(self):
        self.async_openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        accumulated_text: list[str] = []
        callback_tasks: list[asyncio.Task[ContextPartitionItem | None]] = []
        input_items = context_partition.to_openai_input()
        reasoning = Reasoning(effort=reasoning_effort if reasoning_effort else "none")
        round_text: list[str] = []
        text_yielded = False
        tool_call_rounds = 0
        tool_params: list[ToolParam] | None = (
            [cb.tool_spec.to_openai_function_tool_param() for cb in tool_callbacks.values()]
            if tool_callbacks else None
        )
        if input_items and input_items[0].get("role") == "developer":
            logger.info(f"LLM developer message:\n{input_items[0].get('content', '')}")
        async for event in await self.async_openai.responses.create(  # type: ignore
                model=model,
                input=input_items,
                tools=tool_params,
                reasoning=reasoning,
                stream=True,
        ):
            str_to_yield = await self.handle_response_stream_event(
                callback_tasks,
                context_partition,
                event,
                tool_callbacks,
                accumulated_text=accumulated_text,
                model=model,
                ndjson=stream_ndjson,
                on_usage=on_usage,
                round_text=round_text,
            )
            if str_to_yield:
                text_yielded = True
                yield str_to_yield
        while callback_tasks:
            callback_results = await asyncio.gather(*callback_tasks)
            callback_tasks = []
            if all(item is not None for item in callback_results):
                tool_call_rounds += 1
                if max_tool_call_rounds is not None and tool_call_rounds >= max_tool_call_rounds:
                    logger.warning("Reached max_tool_call_rounds=%d, stopping", max_tool_call_rounds)
                    break
                context_partition.extend(result for result in callback_results if result is not None)
                if text_yielded:
                    sep = "\n"
                    accumulated_text.append(sep)
                    yield (json.dumps({"text_delta": sep}) + "\n") if stream_ndjson else sep
                text_yielded = False
                round_text.clear()
                async for event in await self.async_openai.responses.create(  # type: ignore
                        model=model,
                        input=context_partition.to_openai_input(),
                        tools=tool_params,
                        reasoning=reasoning,
                        stream=True,
                ):
                    str_to_yield = await self.handle_response_stream_event(
                        callback_tasks,
                        context_partition,
                        event,
                        tool_callbacks,
                        accumulated_text=accumulated_text,
                        model=model,
                        ndjson=stream_ndjson,
                        on_usage=on_usage,
                        round_text=round_text,
                    )
                    if str_to_yield:
                        text_yielded = True
                        yield str_to_yield

        if accumulated_text:
            context_partition.append(ContextPartitionItem(
                role="assistant", content="".join(accumulated_text)
            ))
