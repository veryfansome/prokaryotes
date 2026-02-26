import json
import logging
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.llm_v1 import FunctionToolCallback
from prokaryotes.search_v1 import PersonDoc, SearchClient

logger = logging.getLogger(__name__)

class SaveUserContextFunctionToolCallback(FunctionToolCallback):
    def __init__(self, person_doc: PersonDoc, search_client: SearchClient):
        self.person_doc = person_doc
        self.search_client = search_client

    async def call(self, arguments: str, call_id: str) -> None:
        try:
            arguments = json.loads(arguments)
            if "context_summary" in arguments and arguments["context_summary"]:
                await self.search_client.add_user_person_fact_doc(self.person_doc, arguments["context_summary"])
            else:
                logging.warning(f"Missing or empty context_summary user {self.person_doc.user_id} in {arguments}")
        except Exception:
            logging.exception(f"Failed to save user {self.person_doc.user_id} context")
        return None  # No continuation

class SearchEmailFunctionToolCallback(FunctionToolCallback):
    def __init__(self, search_client: SearchClient):
        self.search_client = search_client

    async def call(self, arguments: str, call_id: str) -> FunctionCallOutput:
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output='{"messages": []}'
        )
