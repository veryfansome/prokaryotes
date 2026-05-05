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
            tool_callbacks: dict[str, FunctionToolCallback] | None,
            round_text: list[str] | None = None,
            tool_preamble: list[str] | None = None,
            tool_call_seen: list[bool] | None = None,
            emit_text_immediately: bool = False,
            model: str = "",
            ndjson: bool = False,
            on_usage: Callable[[int, int], None] | None = None,
    ) -> str | None:
        if event.type == "response.output_text.delta":
            if round_text is not None:
                round_text.append(event.delta)
            if tool_preamble is not None:
                tool_preamble.append(event.delta)
            if emit_text_immediately:
                return (json.dumps({"text_delta": event.delta}) + "\n") if ndjson else event.delta
        elif event.type == "response.output_text.done":
            pass  # assistant item appended after all rounds complete
        elif event.type == "response.output_item.done" and event.item.type == "function_call":
            logger.info(f"Invoking callback {event.item.name} with arguments {event.item.arguments}")
            preamble = "".join(tool_preamble) if tool_preamble else None
            if tool_preamble is not None:
                tool_preamble.clear()
            if tool_call_seen is not None:
                tool_call_seen[0] = True
            item = ContextPartitionItem(**event.item.__dict__)
            if preamble:
                item.text_preamble = preamble
            context_window.append(item)
            if tool_callbacks and event.item.name in tool_callbacks:
                callback_task: asyncio.Task[ContextPartitionItem | None] = asyncio.create_task(
                    tool_callbacks[event.item.name].call(
                        event.item.arguments,
                        event.item.call_id,
                    )
                )
                callback_tasks.append(callback_task)
            else:
                logger.warning("No callback for tool %r", event.item.name)
            if ndjson:
                return json.dumps({
                    "tool_call": {
                        "name": item.name,
                        "arguments": item.arguments or "{}",
                    }
                }) + "\n"
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
        answer_text: list[str] = []
        input_items = context_partition.to_openai_input()
        reasoning = Reasoning(effort=reasoning_effort if reasoning_effort else "none")
        tool_call_rounds = 0
        tool_params: list[ToolParam] | None = (
            [cb.tool_spec.to_openai_function_tool_param() for cb in tool_callbacks.values()]
            if tool_callbacks else None
        )
        if input_items and input_items[0].get("role") == "developer":
            logger.info(f"LLM developer message:\n{input_items[0].get('content', '')}")
        while True:
            callback_tasks: list[asyncio.Task[ContextPartitionItem | None]] = []
            round_text: list[str] = []
            tool_preamble: list[str] = []
            tool_call_seen = [False]
            round_input = input_items if tool_call_rounds == 0 else context_partition.to_openai_input()
            async for event in await self.async_openai.responses.create(  # type: ignore
                    model=model,
                    input=round_input,
                    tools=tool_params,
                    reasoning=reasoning,
                    stream=True,
            ):
                str_to_yield = await self.handle_response_stream_event(
                    callback_tasks,
                    context_partition,
                    event,
                    tool_callbacks,
                    round_text=round_text,
                    tool_preamble=tool_preamble,
                    tool_call_seen=tool_call_seen,
                    emit_text_immediately=not stream_ndjson,
                    model=model,
                    ndjson=stream_ndjson,
                    on_usage=on_usage,
                )
                if str_to_yield:
                    yield str_to_yield

            round_output = "".join(round_text)
            if tool_call_seen[0]:
                if stream_ndjson and round_output:
                    yield json.dumps({"progress_message": round_output}) + "\n"
                if not callback_tasks:
                    break
                callback_results = await asyncio.gather(*callback_tasks)
                if not all(item is not None for item in callback_results):
                    break
                tool_call_rounds += 1
                if max_tool_call_rounds is not None and tool_call_rounds >= max_tool_call_rounds:
                    logger.warning("Reached max_tool_call_rounds=%d, stopping", max_tool_call_rounds)
                    break
                context_partition.extend(result for result in callback_results if result is not None)
                if round_output and not stream_ndjson:
                    yield "\n"
                continue

            answer_text.extend(round_text)
            if stream_ndjson:
                for delta in round_text:
                    yield json.dumps({"text_delta": delta}) + "\n"
            break

        if answer_text:
            context_partition.append(ContextPartitionItem(
                role="assistant", content="".join(answer_text)
            ))
