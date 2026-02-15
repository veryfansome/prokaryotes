import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator, Callable
from typing import Any

from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseStreamEvent,
    ResponseTextConfigParam,
    ToolParam,
)
from openai.types.responses.response_create_params import ToolChoice
from openai.types.shared_params import Reasoning

from prokaryotes.api_v1.models import (
    ContextPartition,
    ContextPartitionItem,
    FunctionToolCallback,
)

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.async_openai: AsyncOpenAI | None = None

    def init_client(self):
        self.async_openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def close(self):
        await self.async_openai.close()

    async def create_response(
            self,
            context_window: ContextPartition,
            model: str,
            reasoning_effort: str = None,
            stream: bool = False,
            text: ResponseTextConfigParam = None,
            tool_choice: ToolChoice = "auto",
            tool_params: list[ToolParam] = None,
    ):
        return await self.async_openai.responses.create(  # type: ignore
            model=model,
            input=[item.model_dump(exclude_none=True) for item in context_window.items],
            text=text,
            tools=tool_params if tool_params else None,
            tool_choice=tool_choice,
            reasoning=Reasoning(effort=reasoning_effort if reasoning_effort else "none"),
            stream=stream,
        )

    @classmethod
    async def handle_response_stream_event(
            cls,
            event: ResponseStreamEvent,
            context_window: ContextPartition,
            callback_tasks: list[asyncio.Task[ContextPartitionItem | None]],
            tool_callbacks: dict[str, FunctionToolCallback],
            log_events: bool = False,
            ndjson: bool = False,
            on_usage: Callable[[int, int], None] | None = None,
    ) -> str | None:
        if event.type == "response.output_text.delta":
            return (json.dumps({"text_delta": event.delta}) + "\n") if ndjson else event.delta
        elif event.type == "response.output_text.done":
            context_window.append(ContextPartitionItem(role="assistant", content=event.text))
        elif event.type == "response.output_item.done" and event.item.type == "function_call":
            logger.info(f"Invoking callback {event.item.name} with arguments {event.item.arguments}")
            context_window.append(ContextPartitionItem(**event.item.__dict__))
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
        elif (event.type.startswith("response.web_search_call")
              or (event.type == "response.output_item.done" and event.item.type == "search")):
            logger.info(event)
        elif log_events:
            logger.info(event)
        return None

    async def stream_response(
            self,
            context_partition: ContextPartition,
            model: str,
            log_events: bool = False,
            max_tool_call_rounds: int = None,
            on_usage: Callable[[int, int], None] | None = None,
            reasoning_effort: str = None,
            stream_ndjson: bool = False,
            text: ResponseTextConfigParam = None,
            tool_callbacks: dict[str, FunctionToolCallback] = None,
            tool_choice: ToolChoice = "auto",
    ) -> AsyncGenerator[str, Any]:
        tool_params = (
            [cb.tool_spec.to_openai_function_tool_param() for cb in tool_callbacks.values()]
            if tool_callbacks else None
        )
        callback_tasks: list[asyncio.Task[ContextPartitionItem | None]] = []
        text_yielded = False
        tool_call_rounds = 0
        async for event in await self.create_response(
                context_partition, model,
                reasoning_effort=reasoning_effort,
                stream=True,
                text=text,
                tool_choice=tool_choice,
                tool_params=tool_params,
        ):
            str_to_yield = await self.handle_response_stream_event(
                event, context_partition, callback_tasks, tool_callbacks,
                log_events=log_events,
                ndjson=stream_ndjson,
                on_usage=on_usage,
            )
            if str_to_yield:
                text_yielded = True
                yield str_to_yield
        while callback_tasks:
            callback_results = await asyncio.gather(*callback_tasks)
            callback_tasks = []
            # Continuation will fail if all requested function call results are not returned. All FunctionToolCallback
            # calls *MUST* return something if continuation is required.
            if all(item is not None for item in callback_results):
                tool_call_rounds += 1
                if max_tool_call_rounds is not None and tool_call_rounds >= max_tool_call_rounds:
                    logger.warning("Reached max_tool_call_rounds=%d, stopping", max_tool_call_rounds)
                    break
                context_partition.extend(result for result in callback_results if result is not None)
                if text_yielded:
                    yield (json.dumps({"text_delta": "\n"}) + "\n") if stream_ndjson else "\n"
                text_yielded = False
                async for event in await self.create_response(
                        context_partition, model,
                        reasoning_effort=reasoning_effort,
                        stream=True,
                        text=text,
                        tool_choice=tool_choice,
                        tool_params=tool_params,
                ):
                    str_to_yield = await self.handle_response_stream_event(
                        event, context_partition, callback_tasks, tool_callbacks,
                        log_events=log_events,
                        ndjson=stream_ndjson,
                        on_usage=on_usage,
                    )
                    if str_to_yield:
                        text_yielded = True
                        yield str_to_yield
