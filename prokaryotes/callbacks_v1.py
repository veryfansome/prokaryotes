import json
import logging
from elasticsearch import AsyncElasticsearch
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.llm_v1 import FunctionToolCallback
from prokaryotes.search_v1 import PersonDoc

class SaveUserContextFunctionToolCallback(FunctionToolCallback):
    def __init__(self, person_doc: PersonDoc, search_client: AsyncElasticsearch):
        self.person_doc = person_doc
        self.search_client = search_client

    async def call(self, arguments: str, call_id: str) -> None:
        try:
            arguments = json.loads(arguments)
            if "context_summary" in arguments and arguments["context_summary"]:
                self.person_doc.facts.extend(arguments["context_summary"])
            else:
                logging.warning(f"Missing or empty context_summary user {self.person_doc.user_id} in {arguments}")
        except Exception:
            logging.exception(f"Failed to save user {self.person_doc.user_id} context")
        return None  # No continuation

class SearchEmailFunctionToolCallback(FunctionToolCallback):
    def __init__(self, search_client: AsyncElasticsearch):
        self.search_client = search_client

    async def call(self, arguments: str, call_id: str) -> FunctionCallOutput:
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output='{"messages": []}'
        )
