import asyncio
import json
import logging
from openai.types.responses import (
    FunctionToolParam,
    ResponseFormatTextJSONSchemaConfigParam,
    ResponseTextConfigParam,
)

from prokaryotes.llm_v1 import (
    FunctionToolCallback,
    LLMClient,
)
from prokaryotes.models_v1 import (
    FactDoc,
    PersonContext,
    PromptContext,
    TextEmbeddingPrompt,
    TextEmbeddingRequest,
)
from prokaryotes.observer_v1.base import Observer
from prokaryotes.search_v1 import SearchClient
from prokaryotes.utils_v1.context_utils import developer_message_parts
from prokaryotes.utils_v1.text_utils import get_text_embeddings

logger = logging.getLogger(__name__)

class SaveUserFactsFunctionToolCallback(FunctionToolCallback):
    def __init__(self, user_context: PersonContext, search_client: SearchClient):
        self.saved_facts = []
        self.search_client = search_client
        self.user_context = user_context

    async def call(self, arguments: str, call_id: str) -> None:
        try:
            arguments: dict[str, list[str]] = json.loads(arguments)
            if "facts" in arguments and arguments["facts"]:
                normalized_candidates = []
                for candidate in arguments["facts"]:
                    candidate = " ".join(candidate.strip(" .!?\r\n").split())
                    if candidate:
                        normalized_candidates.append(candidate)
                len_before_dedupe = len(normalized_candidates)
                # Exact-duplicate filtering
                existing_fact_texts = {fact.text.casefold() for fact in self.user_context.facts}
                candidates_after_exact_dedupe = [
                    candidate for candidate in normalized_candidates
                    if candidate.casefold() not in existing_fact_texts
                ]
                len_after_exact_dedupe = len(candidates_after_exact_dedupe)
                logger.info(
                    f"Filtered out {len_before_dedupe - len_after_exact_dedupe} candidate facts after exact dedupe"
                )
                # Near-duplicate filtering
                if candidates_after_exact_dedupe:
                    embs_resp = await get_text_embeddings(
                        TextEmbeddingRequest(batch_size=16, prompt=TextEmbeddingPrompt.DOCUMENT,
                                             texts=candidates_after_exact_dedupe,
                                             truncate_to=256)
                    )
                    search_tasks = []
                    for idx, fact in enumerate(candidates_after_exact_dedupe):
                        search_tasks.append(asyncio.create_task(
                            self.search_client.knn_dedupe_facts(
                                about=f"user:{self.user_context.user_id}",
                                match_emb=embs_resp.embs[idx],
                                score_threshold=0.95
                            )
                        ))
                    search_results = await asyncio.gather(*search_tasks)
                    candidates_to_index = []
                    candidate_embs = []
                    for idx, results in enumerate(search_results):
                        if not results:
                            candidates_to_index.append(candidates_after_exact_dedupe[idx])
                            candidate_embs.append(embs_resp.embs[idx])
                        else:
                            # TODO: Pass near-duplicates to an LLM judge
                            # TODO: Alternatively, consolidate near-duplicates into the same fact doc
                            logger.info(
                                f"Filtering out fact candidate '{candidates_after_exact_dedupe[idx]}'"
                                f" as a near-duplicate of {[f.text for f in results]}"
                            )
                    len_after_knn_dedupe = len(candidates_to_index)
                    logger.info(
                        f"Filtered out {len_after_exact_dedupe - len_after_knn_dedupe} candidate facts after knn dedupe"
                    )
                    self.saved_facts = await self.search_client.index_facts(
                        [f"user:{self.user_context.user_id}"],
                        candidates_to_index, candidate_embs,
                    )
            else:
                logging.warning(f"Missing or empty facts in {arguments} (user {self.user_context.user_id})")
        except Exception:
            logging.exception(f"Failed to save user {self.user_context.user_id} facts")
        return None  # No continuation

class UserFactsSavingObserver(Observer):
    def __init__(
            self,
            prompt_context: PromptContext,
            user_context: PersonContext,
            llm_client: LLMClient,
            search_client: SearchClient,
            **kwargs
    ):
        super().__init__(llm_client, **kwargs)
        self.prompt_context = prompt_context
        self.user_context = user_context
        self.save_user_facts_callback = SaveUserFactsFunctionToolCallback(user_context, search_client)

    def developer_message(self) -> str | None:
        message_parts = developer_message_parts(self.prompt_context, self.user_context)
        message_parts.append("---")
        message_parts.append("## Instructions")
        message_parts.append(
            "- Analyze the most recent message. If the user has revealed novel information about themselves,"
            " call the `save_user_facts` function tool to add this information to the \"User info\" section."
        )
        message_parts.append(
            "- When saving a fact that reference the assistant, always maintain a second-person perspective and"
            " address them directly as \"you\". e.g. \"The user is your creator.\" Frame facts as they relate to your"
            " interactions with the user."
        )
        return "\n".join(message_parts)

    async def get_saved_facts(self) -> list[FactDoc]:
        try:
            if self.bg_task:
                await self.bg_task
                return self.save_user_facts_callback.saved_facts
        except Exception:
            logger.exception("Failed to get saved facts")
        return []

    def reasoning_effort(self) -> str:
        # Supported values are `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`.
        fact_cnt = len(self.user_context.facts)
        if fact_cnt <= 10:
            return "none"
        elif fact_cnt <= 20:
            return "low"
        elif fact_cnt <= 30:
            return "medium"
        else:
            return "high"

    def text_param(self) -> ResponseTextConfigParam:
        return ResponseTextConfigParam(format=ResponseFormatTextJSONSchemaConfigParam(
            name="tool_called",
            type="json_schema",
            schema={
                "type": "object",
                "properties": {
                    "tool_called": {
                        "type": "string",
                        "description": "True, if a function tool was called, else False.",
                        "enum": ["True", "False"],
                    },
                },
                "additionalProperties": False,
                "required": ["tool_called"],
            },
            strict=True,
        ))

    def tool_callbacks(self) -> dict[str, FunctionToolCallback]:
        return {"save_user_facts": self.save_user_facts_callback}

    def tool_params(self) -> list[FunctionToolParam]:
        return [
            FunctionToolParam(
                type="function",
                name="save_user_facts",
                description=(
                    "Add new user facts to the \"User info\" section."
                    " Call this function whenever the user mentions private information that you haven't been train"
                    " on and can't looked up."

                    " This includes knowledge about the user or their personal life, including:"
                    " family, friends, colleagues, past events, opinions and preferences,"
                    " hobbies, goals, projects, and more."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "facts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "A flat list of atomic facts, stating information about the user in the"
                                " simplest language possible."
                            ),
                        },
                    },
                    "additionalProperties": False,
                    "required": ["facts"],
                },
                strict=True,
            )
        ]
