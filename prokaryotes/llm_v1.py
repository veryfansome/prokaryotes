import logging
import os
from openai import AsyncOpenAI
from openai.types.responses import ToolParam
from typing import Any, AsyncGenerator, Protocol

from prokaryotes.models_v1 import ChatMessage

logger = logging.getLogger(__name__)

class LLM(Protocol):
    def stream_response(
            self,
            messages: list[ChatMessage],
            model: str,
            reasoning_effort: str = None,
            tool_spec: list[ToolParam] = None,
    ) -> AsyncGenerator[str, Any]:
        pass

class OpenAIClient(LLM):

    def __init__(self, openai_api_key: str):
        self.async_openai = AsyncOpenAI(api_key=openai_api_key)

    async def stream_response(
            self,
            messages: list[ChatMessage],
            model: str,
            reasoning_effort: str = None,
            tool_spec: list[ToolParam] = None,
    ):
        reasoning_config = {"effort": reasoning_effort if reasoning_effort else "none"}
        response = await self.async_openai.responses.create(
            model=model,
            input=[m.model_dump() for m in messages],
            tools=tool_spec if tool_spec else None,
            reasoning=reasoning_config,
            stream=True,
        )
        async for event in response:
            if event.type == "response.output_text.delta":
                yield event.delta
            elif event.type == "response.output_item.done" and event.item.type == "function_call":
                logger.info((event.item.name, event.item.arguments))
                # TODO: implement function calling and continuation
                #async for chunk in self.stream_response(
                #        messages, model,
                #        reasoning_effort=reasoning_effort,
                #        tool_spec=tool_spec,
                #):
                #    yield chunk
            elif event.type.startswith("response.web_search_call"):
                logger.info(event)
            else:
                logger.debug(event)

def get_llm() -> LLM:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        return OpenAIClient(openai_api_key)
    raise RuntimeError("Unable to initialize any LLMs")
