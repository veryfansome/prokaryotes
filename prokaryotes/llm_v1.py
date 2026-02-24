import asyncio
import logging
import os
from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseFunctionToolCall as OpenAIResponseFunctionToolCall,
    ResponseStreamEvent as OpenAIResponseStreamEvent,
    ToolParam as OpenAIToolParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput as OpenAIFunctionCallOutput
from typing import Any, AsyncGenerator, Protocol, is_typeddict

from prokaryotes.models_v1 import ChatMessage

logger = logging.getLogger(__name__)

class FunctionToolCallback(Protocol):
    async def call(self, arguments: str, call_id: str) -> OpenAIFunctionCallOutput:
        pass

class LLMClient(Protocol):
    async def get_response(
            self,
            messages: list[ChatMessage],
            model: str,
            reasoning_effort: str = None,
            tool_callbacks: dict[str, FunctionToolCallback] = None,
            tool_params: list[OpenAIToolParam] = None,
    ) -> str | None:
        return "[BUG: get_response not implemented]"

    async def stream_response(
            self,
            messages: list[ChatMessage],
            model: str,
            reasoning_effort: str = None,
            tool_callbacks: dict[str, FunctionToolCallback] = None,
            tool_params: list[OpenAIToolParam] = None,
    ) -> AsyncGenerator[str, Any]:
        yield "[BUG: stream_response not implemented]"

class OpenAIClient(LLMClient):
    def __init__(self, openai_api_key: str):
        self.async_openai = AsyncOpenAI(api_key=openai_api_key)

    async def create_response(
            self,
            messages: list[ChatMessage | OpenAIFunctionCallOutput | OpenAIResponseFunctionToolCall],
            model: str,
            reasoning_effort: str = None,
            stream: bool = False,
            tool_params: list[OpenAIToolParam] = None,
    ):
        reasoning_config = {"effort": reasoning_effort if reasoning_effort else "none"}
        return await self.async_openai.responses.create(
            model=model,
            input=[(m if (is_typeddict(m) or not isinstance(m, ChatMessage)) else m.model_dump()) for m in messages],
            tools=tool_params if tool_params else None,
            reasoning=reasoning_config,
            stream=stream,
        )

    async def get_response(
            self,
            messages: list[ChatMessage],
            model: str,
            reasoning_effort: str = None,
            tool_callbacks: dict[str, FunctionToolCallback] = None,
            tool_params: list[OpenAIToolParam] = None,
    ) -> str | None:
        response = await self.create_response(
                messages, model,
                reasoning_effort=reasoning_effort,
                tool_params=tool_params,
        )
        if response.output[0].type == "output_text":
            return response.output[0].content[0].text
        elif response.output[0].type == "function_call":
            logger.info((response.output[0].name, response.output[0].arguments))
            # TODO: implement function calling and continuation
        else:
            logger.debug(response)

    @classmethod
    async def handle_response_stream_event(
            cls,
            event: OpenAIResponseStreamEvent,
            messages: list[ChatMessage | OpenAIFunctionCallOutput | OpenAIResponseFunctionToolCall],
            callback_tasks: list[asyncio.Task],
            tool_callbacks: dict[str, FunctionToolCallback],
    ):
        if event.type == "response.output_item.done" and event.item.type == "function_call":
            logger.info(f"Invoking callback {event.item.name} with arguments {event.item.arguments}")
            messages.append(event.item)
            callback_task = asyncio.create_task(tool_callbacks[event.item.name].call(
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
            tool_callbacks: dict[str, FunctionToolCallback] = None,
            tool_params: list[OpenAIToolParam] = None,
    ) -> AsyncGenerator[str, Any]:
        # TODO: Optional pre create_response hooks that can be used for recall and to inject new contexts
        # TODO: Recall unanswered questions for the user, if any, from Neo4j
        callback_tasks = []
        async for event in await self.create_response(
                messages, model,
                reasoning_effort=reasoning_effort,
                stream=True,
                tool_params=tool_params,
        ):
            if event.type == "response.output_text.delta":
                yield event.delta
            else:
                await self.handle_response_stream_event(event, messages, callback_tasks, tool_callbacks)
        while callback_tasks:
            callback_results = await asyncio.gather(*callback_tasks)
            messages.extend(callback_results)
            callback_tasks = []
            async for event in await self.create_response(
                    messages, model,
                    reasoning_effort=reasoning_effort,
                    stream=True,
                    tool_params=tool_params,
            ):
                if event.type == "response.output_text.delta":
                    yield event.delta
                else:
                    await self.handle_response_stream_event(event, messages, callback_tasks, tool_callbacks)

def get_llm_client() -> LLMClient:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return OpenAIClient(openai_api_key)
    raise RuntimeError("Unable to initialize any LLMs")
