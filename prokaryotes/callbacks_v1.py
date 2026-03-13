import asyncio
import json
import logging

from prokaryotes.llm_v1 import FunctionToolCallback
from prokaryotes.models_v1 import (
    ChatMessage,
    TextEmbeddingPrompt,
    TextEmbeddingRequest,
)
from prokaryotes.search_v1 import PersonContext, SearchClient
from prokaryotes.utils_v1.text_utils import get_text_embeddings

logger = logging.getLogger(__name__)

class SaveUserFactsFunctionToolCallback(FunctionToolCallback):
    def __init__(self, user_context: PersonContext, search_client: SearchClient):
        self.saved_facts = []
        self.search_client = search_client
        self.user_context = user_context

    async def call(self, messages: list[ChatMessage], arguments: str, call_id: str) -> None:
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
                        candidates_after_exact_dedupe, embs_resp.embs,
                    )
            else:
                logging.warning(f"Missing or empty facts in {arguments} (user {self.user_context.user_id})")
        except Exception:
            logging.exception(f"Failed to save user {self.user_context.user_id} facts")
        return None  # No continuation

