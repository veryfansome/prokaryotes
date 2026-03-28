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

class FactSavingFunctionCallback(FunctionToolCallback):
    def __init__(self, about: str | None, recalled_facts: list[FactDoc], search_client: SearchClient):
        self.about = about
        self.recalled_facts = recalled_facts
        self.saved_facts = []
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
                len_before_dedupe = len(normalized_candidates)
                # Exact-duplicate filtering
                existing_fact_texts = {fact.text.casefold() for fact in self.recalled_facts}
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
                                match_emb=embs_resp.embs[idx],
                                about=self.about,
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
                        [self.about] if self.about else [],
                        candidates_to_index, candidate_embs,
                    )
            else:
                logging.warning(f"Missing or empty facts in {arguments}")
        except Exception:
            logging.exception("Failed to save facts")
        return None  # No continuation

class FactSavingObserver(Observer):
    def __init__(
            self,
            prompt_context: PromptContext,
            user_context: PersonContext,
            general_facts: list[FactDoc],
            llm_client: LLMClient,
            search_client: SearchClient,
            **kwargs
    ):
        super().__init__(llm_client, **kwargs)
        self.prompt_context = prompt_context
        self.general_facts = general_facts
        self.user_context = user_context
        self.save_general_facts_callback = FactSavingFunctionCallback(
            None, general_facts, search_client
        )
        self.save_user_facts_callback = FactSavingFunctionCallback(
            f"user:{self.user_context.user_id}", user_context.facts, search_client
        )

    def developer_message(self) -> str | None:
        message_parts = developer_message_parts(self.prompt_context, self.user_context, self.general_facts)
        message_parts.append("---")
        message_parts.append("## Instructions")
        message_parts.append(
            "- Analyze the the most recent user message and choose an appropriate tool, whenever applicable, to save"
            " new information you encounter."
        )
        message_parts.append(
            "- When referencing the assistant, always maintain a second-person perspective and address them directly"
            " as \"you\". e.g. \"The user is your creator.\" Frame facts as they relate to your interactions with"
            " the user."
        )
        message_parts.append(
            "- Save new facts only. Do not duplicate or paraphrase facts already in the \"General Info\" and"
            " \"User Info\" sections."
        )
        message_parts.append(
            "- Aim for less than 8 words but you can use more if important context would be lost otherwise."
        )
        message_parts.append(
            "- Don't use more than 20 words."
        )
        return "\n".join(message_parts)

    async def get_saved_facts(self) -> list[FactDoc]:
        try:
            if self.bg_task:
                await self.bg_task
            return [
                *self.save_general_facts_callback.saved_facts,
                *self.save_user_facts_callback.saved_facts,
            ]
        except Exception:
            logger.exception("Failed to get saved facts")
        return []

    def reasoning_effort(self) -> str:
        # Supported values are `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`.
        fact_cnt = len(self.user_context.facts)
        if fact_cnt <= 15:
            return "none"
        elif fact_cnt <= 30:
            return "low"
        elif fact_cnt <= 60:
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
        return {
            "save_general_facts": self.save_general_facts_callback,
            "save_user_facts": self.save_user_facts_callback,
        }

    def tool_params(self) -> list[FunctionToolParam]:
        return [
            FunctionToolParam(
                type="function",
                name="save_general_facts",
                description=(
                    "Add new facts to the \"General info\" section."
                    " Use this tool whenever you encounter new information that you haven't been trained"
                    " on, and which doesn't fall under the scope of the `save_user_facts` tool."

                    " This includes information about you the assistant, your code or runtime environment, or the"
                    " greater world you inhabit."

                    " Use this tool to document the things and situations you encounter, as well as your actions"
                    " and their consequences."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "facts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "A flat list of atomic facts, stating the information you want to save in the"
                                " simplest language possible."
                            ),
                        },
                    },
                    "additionalProperties": False,
                    "required": ["facts"],
                },
                strict=True,
            ),
            FunctionToolParam(
                type="function",
                name="save_user_facts",
                description=(
                    "Add new facts to the \"User info\" section."
                    " Call this tool whenever the user mentions private information."

                    " This includes information about them or their personal life, including:"
                    " their family, friends, colleagues or other acquaintances, their past events and expereinces,"
                    " their opinions and preferences, their interests and hobbies, their goals, projects and"
                    " resolutions, and anything else that would facilitate richer, more personalized interactions."
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
            ),
        ]
