import logging
import os
from datetime import datetime
from openai import AsyncOpenAI
from openai.types.responses import FunctionToolParam, WebSearchToolParam
from typing import Any, AsyncGenerator, Protocol
from zoneinfo import ZoneInfo

from prokaryotes.models_v1 import ChatMessage

logger = logging.getLogger(__name__)

class LLM(Protocol):
    def stream_response(
            self,
            messages: list[ChatMessage],
            time_zone: str = None,
    ) -> AsyncGenerator[str, Any]:
        pass

class OpenAIClient(LLM):

    tools_spec = [
        FunctionToolParam(
            type="function",
            name="get_horoscope",
            description="Get today's horoscope for an astrological sign.",
            parameters={
                "type": "object",
                "properties": {
                    "sign": {
                        "type": "string",
                        "description": "An astrological sign like Taurus or Aquarius",
                    },
                },
                "additionalProperties": False,
                "required": ["sign"],
            },
            strict=True,
        ),
        WebSearchToolParam(
            type="web_search",
            filters={
                "allowed_domains": [
                    "en.wikipedia.org"
                ]
            }
        )
    ]

    @classmethod
    def developer_message(cls, time_zone: str = None):
        time_zone = ZoneInfo("UTC" if not time_zone else time_zone)
        return f"""
Current time: {datetime.now(tz=time_zone).strftime("%Y-%m-%d %H:%M")} {time_zone}
""".strip()

    def __init__(self, openai_api_key: str):
        self.async_openai = AsyncOpenAI(api_key=openai_api_key)

    async def stream_response(self, messages: list[ChatMessage], time_zone: str = None):
        context_window = [{
            "role": "developer",
            "content": self.developer_message(time_zone=time_zone),
        }]
        # TODO: roll long contexts off but in a way that can be recalled
        context_window.extend(m.model_dump() for m in messages)
        response = await self.async_openai.responses.create(
            model="gpt-5",
            input=context_window,
            tools=self.tools_spec,
            stream=True,
        )
        async for event in response:
            logger.debug(event)
            if event.type == "response.output_text.delta":
                yield event.delta
            elif event.type == "response.output_item.done" and event.item.type == "function_call":
                logger.info((event.item.name, event.item.arguments))

def get_llm() -> LLM:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        return OpenAIClient(openai_api_key)
    raise RuntimeError("Unable to initialize any LLMs")
