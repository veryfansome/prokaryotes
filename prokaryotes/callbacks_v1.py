import json
import logging
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.llm_v1 import FunctionToolCallback
from prokaryotes.search_v1 import PersonContext, SearchClient

logger = logging.getLogger(__name__)

class ListDirectoryCallback(FunctionToolCallback):
    async def call(self, arguments: str, call_id: str) -> FunctionCallOutput:
        # TODO: implement actual lookup
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output='{"contents": []}'
        )

# TODO: Maybe it would be better to upload the files to OpenAI?
class ReadFileCallback(FunctionToolCallback):
    async def call(self, arguments: str, call_id: str) -> FunctionCallOutput:
        # TODO: implement actual lookup
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output='{"contents": ""}'
        )

class SaveUserFactsFunctionToolCallback(FunctionToolCallback):
    def __init__(self, user_context: PersonContext, search_client: SearchClient):
        self.user_context = user_context
        self.search_client = search_client

    async def call(self, arguments: str, call_id: str) -> None:
        try:
            arguments: dict[str, list[str]] = json.loads(arguments)
            if "facts" in arguments and arguments["facts"]:
                normalized_candidates = []
                for candidate in arguments["facts"]:
                    candidate = " ".join(candidate.strip(" .!?\r\n").split())
                    if candidate:
                        normalized_candidates.append(candidate)
                # Exact dedupe
                existing_fact_texts = {fact.text.casefold() for fact in self.user_context.facts}
                candidates_after_exact_dedupe = [
                    candidate for candidate in normalized_candidates
                    if candidate.casefold() not in existing_fact_texts
                ]
                # TODO: Additional dedupe via semantic similarity or another pass with an LLM (offline?)
                await self.search_client.index_facts([f"user_{self.user_context.user_id}"], candidates_after_exact_dedupe)
            else:
                logging.warning(f"Missing or empty facts in {arguments} (user {self.user_context.user_id})")
        except Exception:
            logging.exception(f"Failed to save user {self.user_context.user_id} facts")
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
