import logging
from abc import (
    ABC,
    abstractmethod,
)
from datetime import UTC, datetime

from elasticsearch import AsyncElasticsearch

from prokaryotes.models_v1 import ToolCallDoc
from prokaryotes.search_v1.topics import TopicSearcher
from prokaryotes.utils_v1.text_utils import (
    str_similarity_batch,
    strip_punctuation,
    text_to_md5,
)

logger = logging.getLogger(__name__)

tool_call_mappings = {
    "dynamic": "strict",
    "properties": {
        "created_at":      {"type": "date"},
        "labels":          {"type": "keyword"},
        "output":          {"type": "text"},
        "output_hash":     {"type": "keyword"},
        "prompt_summary": {
            "type":            "text",
            "analyzer":        "standard",
            "search_analyzer": "custom_query_analyzer",
        },
        "prompt_summary_emb": {
            "type":       "dense_vector",
            "dims":       256,
            "index":      True,
            "similarity": "cosine",
        },
        "reason": {
            "type":            "text",
            "analyzer":        "standard",
            "search_analyzer": "custom_query_analyzer",
        },
        "reason_emb": {
            "type":       "dense_vector",
            "dims":       256,
            "index":      True,
            "similarity": "cosine",
        },
        "search_keywords":     {"type": "keyword"},
        "tool_arguments":      {"type": "text"},
        "tool_arguments_hash": {"type": "keyword"},
        "tool_name":           {"type": "keyword"},
        "topics":              {"type": "keyword"},
    }
}


class ToolCallSearcher(TopicSearcher, ABC):
    @property
    @abstractmethod
    def es(self) -> AsyncElasticsearch:
        pass

    async def get_tool_call(self, call_id: str) -> ToolCallDoc | None:
        try:
            result = await self.es.get(index="tool-calls", id=call_id)
            return ToolCallDoc(doc_id=result["_id"], **result["_source"]) if result else None
        except Exception:
            logger.exception(f"Failed to retrieve ToolCallDoc(doc_id={call_id})")

    async def index_tool_call(
            self,
            call_id: str,
            labels: list[str],
            output: str,
            prompt_summary: str,
            prompt_summary_emb: list[float],
            reason: str,
            reason_emb: list[float],
            search_keywords: list[str],
            tool_arguments: str,
            tool_name: str,
            topics: list[str],
    ):
        now = datetime.now(UTC)
        output_hash = text_to_md5(output)
        doc = ToolCallDoc(
            doc_id=call_id,
            created_at=now,
            labels=labels,
            output=output,
            output_hash=output_hash,
            reason=reason,
            search_keywords=search_keywords,
            tool_arguments=tool_arguments,
            tool_name=tool_name,
        )
        try:
            await self.es.index(
                index="tool-calls",
                id=call_id,
                document=(doc.model_dump() | {
                    "output_hash": output_hash,
                    "prompt_summary": prompt_summary,
                    "prompt_summary_emb": prompt_summary_emb,
                    "reason_emb": reason_emb,
                    "tool_arguments_hash": text_to_md5(tool_arguments),
                    "topics": topics,
                }),
            )
            return doc
        except Exception:
            logger.exception(f"Failed to index {doc}")

    async def search_tool_call(
            self,
            candidates: int = 50,
            excluded_ids: list[str] = None,
            knn_num_candidates: int = 100,
            labels_and: list[str] | None = None,
            labels_or: list[str] | None = None,
            limit: int = 3,
            match: str = None,
            match_emb: list[float] = None,
            min_final_score: float | None = None,
            min_initial_score: float = 0.5,
            min_output_similarity_score: float = 0.9,
            not_labels_and: list[str] | None = None,
            not_labels_or: list[str] | None = None,
            prompt_match_boost: float = 0.6,
            reason_match_boost: float = 0.8,
            search_keywords_boost: float = 5.0,
            similar_topics_boost: float = 0.4,
    ) -> list[ToolCallDoc]:
        shared_filters = []
        if labels_and:
            for labels in labels_and:
                shared_filters.append({"term": {"labels": labels}})
        if labels_or:
            shared_filters.append({"terms": {"labels": labels_or}})

        shared_must_not = [{"term": {"labels": "deactivated"}}]
        if excluded_ids:
            # TODO: Excluded docs should be deduped against
            shared_must_not.append({"ids": {"values": excluded_ids}})
        if not_labels_and:
            for labels in not_labels_and:
                shared_must_not.append({"term": {"labels": labels}})
        if not_labels_or:
            shared_must_not.append({"terms": {"labels": not_labels_or}})

        main_query = {
            "filter": shared_filters,
            "must_not": shared_must_not,
        }

        similar_topics = []
        if match and match_emb:
            similar_topics = await self.search_topics(
                match, match_emb,
                min_score=0.75,
            )
        match_tokens = []
        if match:
            match_tokens = [token for token in (strip_punctuation(tok) for tok in match.split()) if token]
            should_clauses = [
                {
                    "match": {
                        "prompt_summary": {
                            "query": match,
                            "boost": prompt_match_boost,
                            "_name": "prompt_summary_match",
                        }
                    }
                },
                {
                    "match": {
                        "reason": {
                            "query": match,
                            "boost": reason_match_boost,
                            "_name": "reason_match",
                        }
                    }
                },
            ]
            if match_tokens:
                should_clauses.append({
                    "terms": {
                        "search_keywords": match_tokens,
                        "boost": search_keywords_boost,
                        "_name": "search_keywords_terms",
                    }
                })
            if similar_topics:
                should_clauses.append({
                    "terms": {
                        "topics": similar_topics,
                        "boost": similar_topics_boost,
                        "_name": "topics_terms",
                    }
                })
            main_query["should"] = should_clauses

        lexical_search_kwargs = {
            "index": "tool-calls",
            "query": {
                "bool": main_query,
            },
            "size": candidates,
        }
        if min_initial_score is not None:
            lexical_search_kwargs["min_score"] = min_initial_score

        lexical_response = await self.es.search(**lexical_search_kwargs)
        lexical_hits = lexical_response["hits"]["hits"]
        logger.debug(f"Search tool call lexical hit count: {len(lexical_hits)}")

        prompt_knn_hits = []
        reason_knn_hits = []
        if match_emb:
            prompt_knn_search_kwargs = {
                "index": "tool-calls",
                "size": candidates,
                "query": {
                    "bool": {
                        "filter": shared_filters,
                        "must_not": shared_must_not,
                    }
                },
                "knn": {
                    "field": "prompt_summary_emb",
                    "query_vector": match_emb,
                    "boost": 1.0,
                    "num_candidates": knn_num_candidates,
                    "k": candidates,
                    "filter": {
                        "bool": {
                            "filter": shared_filters,
                            "must_not": shared_must_not,
                        }
                    }
                }
            }
            prompt_knn_response = await self.es.search(**prompt_knn_search_kwargs)
            prompt_knn_hits = prompt_knn_response["hits"]["hits"]
            logger.debug(f"Search tool call prompt_summary knn hit count: {len(prompt_knn_hits)}")

            reason_knn_search_kwargs = {
                "index": "tool-calls",
                "size": candidates,
                "query": {
                    "bool": {
                        "filter": shared_filters,
                        "must_not": shared_must_not,
                    }
                },
                "knn": {
                    "field": "reason_emb",
                    "query_vector": match_emb,
                    "boost": 1.0,
                    "num_candidates": knn_num_candidates,
                    "k": candidates,
                    "filter": {
                        "bool": {
                            "filter": shared_filters,
                            "must_not": shared_must_not,
                        }
                    }
                }
            }
            reason_knn_response = await self.es.search(**reason_knn_search_kwargs)
            reason_knn_hits = reason_knn_response["hits"]["hits"]
            logger.debug(f"Search tool call reason knn hit count: {len(reason_knn_hits)}")

        hits_by_id = {h["_id"]: h for h in lexical_hits}
        for h in prompt_knn_hits:
            if h["_id"] not in hits_by_id:
                hits_by_id[h["_id"]] = h
        for h in reason_knn_hits:
            if h["_id"] not in hits_by_id:
                hits_by_id[h["_id"]] = h
        hits = list(hits_by_id.values())
        logger.debug(f"Search tool call merged hit count: {len(hits)}")

        lexical_scores = {h["_id"]: h.get("_score", 0.0) for h in lexical_hits}
        prompt_knn_scores = {h["_id"]: h.get("_score", 0.0) for h in prompt_knn_hits}
        reason_knn_scores = {h["_id"]: h.get("_score", 0.0) for h in reason_knn_hits}

        lexical_max = max(lexical_scores.values(), default=0.0)
        # So weak lexical matches without search_keyword hits don’t auto-normalize to 1.0
        lexical_denominator = max(lexical_max, search_keywords_boost)
        lexical_scores_norm = {
            doc_id: (score / lexical_denominator) if lexical_denominator else 0.0
            for doc_id, score in lexical_scores.items()
        }
        prompt_semantic_scores_abs = {
            doc_id: max(0.0, min(1.0, score))
            for doc_id, score in prompt_knn_scores.items()
        }
        reason_semantic_scores_abs = {
            doc_id: max(0.0, min(1.0, score))
            for doc_id, score in reason_knn_scores.items()
        }
        semantic_scores_abs = {
            doc_id: max(
                prompt_semantic_scores_abs.get(doc_id, 0.0),
                reason_semantic_scores_abs.get(doc_id, 0.0),
            )
            for doc_id in (set(prompt_semantic_scores_abs) | set(reason_semantic_scores_abs))
        }
        lexical_keyword_hit_ids = set()
        for h in lexical_hits:
            matched_queries = h.get("matched_queries", [])
            if isinstance(matched_queries, dict):
                query_names = set(matched_queries.keys())
            elif isinstance(matched_queries, list):
                query_names = set(matched_queries)
            else:
                query_names = set()
            if "search_keywords_terms" in query_names:
                lexical_keyword_hit_ids.add(h["_id"])
        non_keyword_lexical_cap = min(0.9, (search_keywords_boost - 1e-3) / lexical_denominator)

        # Penalty-only use of semantics: dissimilar candidates are pushed down.
        dissimilarity_floor = 0.6
        dissimilarity_penalty_weight = 0.5

        deduped_hits = {}
        for h in hits:
            tool_arguments = h["_source"].get("tool_arguments")
            tool_name = h["_source"].get("tool_name")
            lexical_norm = lexical_scores_norm.get(h["_id"], 0.0)
            keyword_hit = h["_id"] in lexical_keyword_hit_ids
            if lexical_scores_norm and not keyword_hit:
                lexical_norm = min(lexical_norm, non_keyword_lexical_cap)
            prompt_semantic_norm = prompt_semantic_scores_abs.get(h["_id"], 0.0)
            reason_semantic_norm = reason_semantic_scores_abs.get(h["_id"], 0.0)
            semantic_norm = semantic_scores_abs.get(h["_id"], 0.0)
            if lexical_scores_norm:
                dissimilarity_penalty = dissimilarity_penalty_weight * max(0.0, dissimilarity_floor - semantic_norm)
                final_score = lexical_norm - dissimilarity_penalty
            else:
                # Fallback: if lexical retrieval found nothing, preserve semantic ordering.
                final_score = semantic_norm
                dissimilarity_penalty = 0.0
            h["_rerank_score"] = final_score
            logger.debug(
                f"ToolCallDoc ID: {h['_id']} | Score: {h['_score']:.4f}"
                f" | lex: {lexical_norm:.4f}"
                f" | sem_prompt: {prompt_semantic_norm:.4f}"
                f" | sem_reason: {reason_semantic_norm:.4f}"
                f" | penalty: {dissimilarity_penalty:.4f}"
                f" | final: {final_score:.4f}"
                f" | keyword: {keyword_hit}"
                f" | tool: {tool_name} | args: {tool_arguments}"
            )
            tool_sig = (tool_name, tool_arguments)
            if tool_sig in deduped_hits:
                existing = deduped_hits[tool_sig]
                existing_score = existing.get("_rerank_score", 0.0)
                if (
                        h["_rerank_score"] > existing_score
                        or (
                                h["_rerank_score"] == existing_score
                                and existing["_source"]["created_at"] < h["_source"].get("created_at")
                        )
                ):
                    deduped_hits[tool_sig] = h
                continue
            deduped_hits[tool_sig] = h
        logger.debug(f"Search tool call post-dedupe hit count: {len(deduped_hits)}")

        sorted_hits = sorted(
            deduped_hits.values(),
            key=lambda hit: (
                hit.get("_rerank_score", 0.0),
                hit["_source"].get("created_at"),
            ),
            reverse=True,
        )

        # Refresh the most recent output for each argument-signature within the same scope.
        shortlist_size = max(limit * 4, 12)
        shortlist_hits = sorted_hits[:shortlist_size]
        refreshed_candidates: list[tuple[ToolCallDoc, float]] = []
        seen_arg_hashes: set[tuple[str, str]] = set()
        for h in shortlist_hits:
            tool_name = h["_source"].get("tool_name")
            tool_arguments_hash = h["_source"].get("tool_arguments_hash")
            arg_key = (tool_name, tool_arguments_hash)
            if arg_key in seen_arg_hashes:
                continue
            seen_arg_hashes.add(arg_key)

            rerank_score = h.get("_rerank_score", 0.0)
            latest_calls = await self.search_tool_call_by_arguments_hash(
                tool_name=tool_name,
                tool_arguments_hash=tool_arguments_hash,
                excluded_ids=excluded_ids,
                labels_and=labels_and,
                labels_or=labels_or,
                not_labels_and=not_labels_and,
                not_labels_or=not_labels_or,
                size=1,
            )
            if latest_calls:
                refreshed_candidates.append((latest_calls[0], rerank_score))
            else:
                refreshed_candidates.append((ToolCallDoc(doc_id=h["_id"], **h["_source"]), rerank_score))
        logger.debug(f"Search tool call refreshed candidate count: {len(refreshed_candidates)}")

        # Output dedupe prefers recency: keep the newest representative among near-identical outputs.
        refreshed_candidates.sort(key=lambda item: item[0].created_at, reverse=True)
        output_deduped: list[tuple[ToolCallDoc, float]] = []
        output_hashes: set[str] = set()
        for doc, rerank_score in refreshed_candidates:
            output_hash = doc.output_hash or text_to_md5(doc.output)
            if output_hash in output_hashes:
                continue
            if output_deduped:
                output_similarities = str_similarity_batch(
                    doc.output,
                    [existing_doc.output for existing_doc, _ in output_deduped],
                )
                if any(score >= min_output_similarity_score for score in output_similarities):
                    continue
            output_hashes.add(output_hash)
            output_deduped.append((doc, rerank_score))
        logger.debug(f"Search tool call post-output-dedupe count: {len(output_deduped)}")

        output_deduped.sort(
            key=lambda item: (
                item[1],
                item[0].created_at,
            ),
            reverse=True,
        )
        if min_final_score is not None:
            output_deduped = [
                item for item in output_deduped
                if item[1] >= min_final_score
            ]
        results = [doc for doc, _ in output_deduped[:limit]]
        logger.debug(f"Search tool call final hit count: {len(results)}")
        return results

    async def search_tool_call_by_arguments_hash(
        self,
        tool_name: str,
        tool_arguments_hash: str,
        excluded_ids: list[str] | None = None,
        labels_and: list[str] | None = None,
        labels_or: list[str] | None = None,
        not_labels_and: list[str] | None = None,
        not_labels_or: list[str] | None = None,
        size: int = None,
    ) -> list[ToolCallDoc]:
        shared_filters = [
            {"term": {"tool_name": tool_name}},
            {"term": {"tool_arguments_hash": tool_arguments_hash}},
        ]
        if labels_and:
            for label in labels_and:
                shared_filters.append({"term": {"labels": label}})
        if labels_or:
            shared_filters.append({"terms": {"labels": labels_or}})

        shared_must_not = [
            {"term": {"labels": "deactivated"}},
        ]
        if excluded_ids:
            shared_must_not.append({"ids": {"values": excluded_ids}})
        if not_labels_and:
            for label in not_labels_and:
                shared_must_not.append({"term": {"labels": label}})
        if not_labels_or:
            shared_must_not.append({"terms": {"labels": not_labels_or}})

        search_kwargs = {
            "index": "tool-calls",
            "query": {
                "bool": {
                    "filter": shared_filters,
                    "must_not": shared_must_not,
                }
            },
            "sort": [{"created_at": {"order": "desc"}}],
        }
        if size:
            search_kwargs["size"] = size
        response = await self.es.search(**search_kwargs)
        hits = response["hits"]["hits"]
        return [ToolCallDoc(doc_id=h["_id"], **h["_source"]) for h in hits]
