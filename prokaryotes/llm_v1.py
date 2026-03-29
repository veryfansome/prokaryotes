import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import (
    Any,
    Protocol,
    is_typeddict,
    runtime_checkable,
)

from openai import AsyncOpenAI
from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseStreamEvent,
    ResponseTextConfigParam,
    ToolParam,
)
from openai.types.responses.response_create_params import ToolChoice
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.shared_params import Reasoning

from prokaryotes.models_v1 import (
    ChatMessage,
    ToolCallDoc,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class FunctionCallOutputIndexer(Protocol):
    async def index(
            self,
            prompt_messages: list[ChatMessage],
            arguments: str,
            output: str,
    ) -> ToolCallDoc | None:
        pass


class FunctionToolCallback(Protocol):
    @property
    def tool_param(self) -> FunctionToolParam:  # type: ignore
        pass

    async def call(
            self,
            arguments: str,
            call_id: str,
    ) -> FunctionCallOutput | None:
        pass


class LLMClient(Protocol):
    async def stream_response(
            self,
            context_window: list[ChatMessage | FunctionCallOutput | ResponseFunctionToolCall],
            model: str,
            log_events: bool = False,
            reasoning_effort: str = None,
            stream_ndjson: bool = False,
            text: ResponseTextConfigParam = None,
            tool_callbacks: dict[str, FunctionToolCallback] = None,
            tool_choice: ToolChoice = "auto",
            tool_params: list[ToolParam] = None,
    ) -> AsyncGenerator[str, Any]:
        yield "[BUG: stream_response not implemented]"


class OpenAIClient(LLMClient):
    def __init__(self, openai_api_key: str):
        self.async_openai = AsyncOpenAI(api_key=openai_api_key)

    async def create_response(
            self,
            context_window: list[ChatMessage | FunctionCallOutput | ResponseFunctionToolCall],
            model: str,
            reasoning_effort: str = None,
            stream: bool = False,
            text: ResponseTextConfigParam = None,
            tool_choice: ToolChoice = "auto",
            tool_params: list[ToolParam] = None,
    ):
        return await self.async_openai.responses.create(  # type: ignore
            model=model,
            include=([
                "web_search_call.action.sources",
                "web_search_call.results",
            ] if any(is_typeddict(param) and param["type"] == "web_search" for param in tool_params) else []),
            input=[
                (m if (is_typeddict(m) or not isinstance(m, ChatMessage)) else m.model_dump())
                for m in context_window
            ],
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
            context_window: list[ChatMessage | FunctionCallOutput | ResponseFunctionToolCall],
            callback_tasks: list[asyncio.Task[FunctionCallOutput | None]],
            tool_callbacks: dict[str, FunctionToolCallback],
            log_events: bool = False,
            ndjson: bool = False,
    ) -> str | None:
        if event.type == "response.output_text.delta":
            return (json.dumps({"text_delta": event.delta}) + "\n") if ndjson else event.delta
        elif event.type == "response.output_text.done":
            context_window.append(ChatMessage(role="assistant", content=event.text))
        elif event.type == "response.output_item.done" and event.item.type == "function_call":
            logger.info(f"Invoking callback {event.item.name} with arguments {event.item.arguments}")
            context_window.append(event.item)
            callback_task: asyncio.Task[FunctionCallOutput | None] = asyncio.create_task(
                tool_callbacks[event.item.name].call(
                    event.item.arguments,
                    event.item.call_id,
                )
            )
            callback_tasks.append(callback_task)
        elif (event.type.startswith("response.web_search_call")
                or (event.type == "response.output_item.done" and event.item.type == "search")):
            logger.info(event)
        elif log_events:
            logger.info(event)
        return None

    async def stream_response(
            self,
            context_window: list[ChatMessage | FunctionCallOutput | ResponseFunctionToolCall],
            model: str,
            log_events: bool = False,
            reasoning_effort: str = None,
            stream_ndjson: bool = False,
            text: ResponseTextConfigParam = None,
            tool_callbacks: dict[str, FunctionToolCallback] = None,
            tool_choice: ToolChoice = "auto",
            tool_params: list[ToolParam] = None,
    ) -> AsyncGenerator[str, Any]:
        callback_tasks: asyncio.Task[FunctionCallOutput | None] = []
        async for event in await self.create_response(
                context_window, model,
                reasoning_effort=reasoning_effort,
                stream=True,
                text=text,
                tool_choice=tool_choice,
                tool_params=tool_params,
        ):
            str_to_yield = await self.handle_response_stream_event(
                event, context_window, callback_tasks, tool_callbacks,
                log_events=log_events,
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
                context_window.extend(result for result in callback_results if result is not None)
                async for event in await self.create_response(
                        context_window, model,
                        reasoning_effort=reasoning_effort,
                        stream=True,
                        text=text,
                        tool_choice=tool_choice,
                        tool_params=tool_params,
                ):
                    str_to_yield = await self.handle_response_stream_event(
                        event, context_window, callback_tasks, tool_callbacks,
                        log_events=log_events,
                        ndjson=stream_ndjson,
                    )
                    if str_to_yield:
                        yield str_to_yield


def get_llm_client() -> LLMClient:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return OpenAIClient(openai_api_key)
    raise RuntimeError("Unable to initialize any LLMs")
