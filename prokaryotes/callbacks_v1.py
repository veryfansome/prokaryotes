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
            arguments: dict[str, list[str]] = json.loads(arguments)
            if "context_summary" in arguments and arguments["context_summary"]:
                normalized_candidates = []
                for candidate in arguments["context_summary"]:
                    candidate = " ".join(candidate.strip(" .!?\r\n").split())
                    if candidate:
                        normalized_candidates.append(candidate)
                # Exact dedupe
                existing_fact_texts = {fact.text.casefold() for fact in self.person_doc.facts}
                candidates_after_exact_dedupe = [
                    candidate for candidate in normalized_candidates
                    if candidate.casefold() not in existing_fact_texts
                ]
                # TODO: Additional dedupe via semantic similarity or another pass with an LLM
                await self.search_client.add_user_person_fact_doc(self.person_doc, candidates_after_exact_dedupe)
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
