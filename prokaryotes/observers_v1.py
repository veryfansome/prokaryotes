from openai.types.responses import FunctionToolParam

from prokaryotes.llm_v1 import FunctionToolCallback, LLMClient
from prokaryotes.models_v1 import ChatMessage

class Observer:
    def __init__(
            self,
            llm_client: LLMClient,
            tool_callbacks: dict[str, FunctionToolCallback],
            tool_params: list[FunctionToolParam],
            model: str = "gpt-5.1",
    ):
        self.llm_client = llm_client
        self.model = model
        self.tool_callbacks = tool_callbacks
        self.tool_params = tool_params

    async def observe(self, messages: list[ChatMessage]):
        await self.llm_client.get_response(
            messages, self.model,
            tool_callbacks=self.tool_callbacks,
            tool_params=self.tool_params,
        )
