from elasticsearch import AsyncElasticsearch
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.llm_v1 import FunctionToolCallback

class SaveUserContextFunctionToolCallback(FunctionToolCallback):
    def __init__(self, search_client: AsyncElasticsearch):
        self.search_client = search_client

    async def call(self, arguments: str, call_id: str) -> FunctionCallOutput:
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output='{"status": "done"}'
        )

class SearchEmailFunctionToolCallback(FunctionToolCallback):
    def __init__(self, search_client: AsyncElasticsearch):
        self.search_client = search_client

    async def call(self, arguments: str, call_id: str) -> FunctionCallOutput:
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output='{"messages": []}'
        )
