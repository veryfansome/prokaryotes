import asyncio
import logging
from openai.types.responses import (
    FunctionToolParam,
    ResponseTextConfigParam,
)

from prokaryotes.llm_v1 import (
    FunctionToolCallback,
    LLMClient,
)
from prokaryotes.models_v1 import ChatMessage
from prokaryotes.utils_v1.logging_utils import log_async_task_exception

logger = logging.getLogger(__name__)

class Observer:
    def __init__(self, llm_client: LLMClient, model: str = "gpt-5.1"):
        self.bg_task = None
        self.llm_client = llm_client
        self.model = model
        self.response_text = ""

    def developer_message(self) -> str | None:
        pass

    async def observe(self, messages: list[ChatMessage]):
        context_window = []
        developer_message = self.developer_message()
        if developer_message:
            logger.debug(f"{self.__class__.__name__} developer message:\n{developer_message}")
            context_window.append(ChatMessage(role="developer", content=developer_message))
        # TODO: Truncate observed window (last 5 messages?)
        context_window.extend(messages)

        async for chunk in self.llm_client.stream_response(
                context_window, self.model,
                reasoning_effort=self.reasoning_effort(),
                text=self.text_param(),
                tool_callbacks=self.tool_callbacks(),
                tool_params=self.tool_params(),
        ):
            self.response_text += chunk
        logger.info(f"{self.__class__.__name__} response text: {self.response_text}")

    def observe_in_background(self, messages: list[ChatMessage]):
        self.bg_task = asyncio.create_task(self.observe(messages))
        self.bg_task.add_done_callback(log_async_task_exception)

    def reasoning_effort(self) -> str:
        return "none"

    def text_param(self) -> ResponseTextConfigParam:
        return ResponseTextConfigParam(verbosity="low")

    def tool_callbacks(self) -> dict[str, FunctionToolCallback]:
        return {}

    def tool_params(self) -> list[FunctionToolParam]:
        return []
