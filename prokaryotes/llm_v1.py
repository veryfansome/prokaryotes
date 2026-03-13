import asyncio
import json
import logging
import os
from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseStreamEvent,
    ResponseTextConfigParam,
    ToolParam,
)
from openai.types.responses import FunctionToolParam
from openai.types.responses.response_input_param import FunctionCallOutput
from typing import Any, AsyncGenerator, Protocol, is_typeddict

from prokaryotes.models_v1 import ChatMessage

logger = logging.getLogger(__name__)

class FunctionToolCallback(Protocol):
    @property
    def tool_param(self) -> FunctionToolParam:
        pass

    async def call(
            self,
            messages: list[ChatMessage],
            arguments: str,
            call_id: str,
    ) -> FunctionCallOutput | None:
        pass

class LLMClient(Protocol):
    async def stream_response(
            self,
            messages: list[ChatMessage],
            model: str,
            reasoning_effort: str = None,
            stream_ndjson: bool = False,
            text: ResponseTextConfigParam = None,
            tool_callbacks: dict[str, FunctionToolCallback] = None,
            tool_params: list[ToolParam] = None,
    ) -> AsyncGenerator[str, Any]:
        yield "[BUG: stream_response not implemented]"

class OpenAIClient(LLMClient):
    def __init__(self, openai_api_key: str):
        self.async_openai = AsyncOpenAI(api_key=openai_api_key)

    async def create_response(
            self,
            messages: list[ChatMessage | FunctionCallOutput | ResponseFunctionToolCall],
            model: str,
            reasoning_effort: str = None,
            stream: bool = False,
            text: ResponseTextConfigParam = None,
            tool_params: list[ToolParam] = None,
    ):
        reasoning_config = {"effort": reasoning_effort if reasoning_effort else "none"}
        return await self.async_openai.responses.create(
            model=model,
            input=[(m if (is_typeddict(m) or not isinstance(m, ChatMessage)) else m.model_dump()) for m in messages],
            text=text,
            tools=tool_params if tool_params else None,
            reasoning=reasoning_config,
            stream=stream,
        )

    @classmethod
    async def handle_response_stream_event(
            cls,
            event: ResponseStreamEvent,
            messages: list[ChatMessage | FunctionCallOutput | ResponseFunctionToolCall],
            callback_tasks: list[asyncio.Task],
            tool_callbacks: dict[str, FunctionToolCallback],
            ndjson: bool = False,
    ) -> str | None:
        if event.type == "response.output_text.delta":
            return (json.dumps({"text_delta": event.delta}) + "\n") if ndjson else event.delta
        elif event.type == "response.output_item.done" and event.item.type == "function_call":
            logger.info(f"Invoking callback {event.item.name} with arguments {event.item.arguments}")
            messages.append(event.item)
            callback_task = asyncio.create_task(tool_callbacks[event.item.name].call(
                messages,
                event.item.arguments,
                event.item.call_id,
            ))
            callback_tasks.append(callback_task)
        elif event.type.startswith("response.web_search_call"):
            logger.info(event)
        else:
            logger.debug(event)

    async def stream_response(
            self,
            messages: list[ChatMessage],
            model: str,
            reasoning_effort: str = None,
            stream_ndjson: bool = False,
            text: ResponseTextConfigParam = None,
            tool_callbacks: dict[str, FunctionToolCallback] = None,
            tool_params: list[ToolParam] = None,
    ) -> AsyncGenerator[str, Any]:
        # TODO: Optional pre create_response hooks that can be used for recall and to inject new contexts
        callback_tasks = []
        async for event in await self.create_response(
                messages, model,
                reasoning_effort=reasoning_effort,
                stream=True,
                text=text,
                tool_params=tool_params,
        ):
            str_to_yield = await self.handle_response_stream_event(
                event, messages, callback_tasks, tool_callbacks,
                ndjson=stream_ndjson,
            )
            if str_to_yield:
                yield str_to_yield
        while callback_tasks:
            callback_results = await asyncio.gather(*callback_tasks)
            callback_tasks = []
            # Continuation will fail if all requested function call results are not returned. All FunctionToolCallback
            # calls *MUST* return something if continuation is required.
            if all(item is not None for item in callback_results):
                messages.extend(result for result in callback_results if result is not None)
                async for event in await self.create_response(
                        messages, model,
                        reasoning_effort=reasoning_effort,
                        stream=True,
                        text=text,
                        tool_params=tool_params,
                ):
                    str_to_yield = await self.handle_response_stream_event(
                        event, messages, callback_tasks, tool_callbacks,
                        ndjson=stream_ndjson,
                    )
                    if str_to_yield:
                        yield str_to_yield

def get_llm_client() -> LLMClient:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return OpenAIClient(openai_api_key)
    raise RuntimeError("Unable to initialize any LLMs")
