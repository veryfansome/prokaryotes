from elasticsearch import AsyncElasticsearch
from neo4j import AsyncDriver
from openai.types.responses import FunctionToolParam

from prokaryotes.callbacks_v1 import SaveUserContextFunctionToolCallback
from prokaryotes.llm_v1 import FunctionToolCallback, LLMClient
from prokaryotes.models_v1 import ChatMessage
from prokaryotes.search_v1 import PersonDoc
from prokaryotes.tool_params_v1 import save_user_context_tool_param

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
        async for _ in self.llm_client.stream_response(
            messages, self.model,
            tool_callbacks=self.tool_callbacks,
            tool_params=self.tool_params,
        ):
            pass  # Drop streamed data

def get_observers(
        person_doc: PersonDoc,
        graph_client: AsyncDriver,
        llm_client: LLMClient,
        search_client: AsyncElasticsearch,
) -> list[Observer]:
    return [
        Observer(
            llm_client=llm_client,
            tool_callbacks={
                save_user_context_tool_param["name"]: SaveUserContextFunctionToolCallback(person_doc, search_client)
            },
            tool_params=[
                save_user_context_tool_param,
            ],
        ),
    ]