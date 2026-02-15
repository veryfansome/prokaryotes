import os
from openai import AsyncOpenAI
from typing import Any, AsyncGenerator, Protocol

from prokaryotes.models_v1 import ChatMessage

class LLM(Protocol):
    def stream_chat_completion_response(self, messages: list[ChatMessage]) -> AsyncGenerator[str, Any]:
        pass

class OpenAIClient(LLM):
    def __init__(self, openai_api_key: str):
        self.async_openai = AsyncOpenAI(api_key=openai_api_key)

    async def stream_chat_completion_response(self, messages: list[ChatMessage]):
        response = await self.async_openai.chat.completions.create(
            model="gpt-4o",
            messages=[m.model_dump() for m in messages],
            stream=True,
        )
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

def get_llm() -> LLM:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        return OpenAIClient(openai_api_key)
    raise RuntimeError("Unable to initialize any LLMs")
