import logging
from abc import (
    ABC,
    abstractmethod,
)
from datetime import UTC, datetime
from typing import Literal

from elasticsearch import AsyncElasticsearch

from prokaryotes.models_v1 import ToolCallDoc
from prokaryotes.search_v1.topics import TopicSearcher
from prokaryotes.utils_v1.text_utils import strip_punctuation, text_to_md5

logger = logging.getLogger(__name__)

tool_call_mappings = {
    "dynamic": "strict",
    "properties": {
        "created_at":      {"type": "date"},
        "dedupe_strategy": {"type": "keyword"},
        "labels":          {"type": "keyword"},
        "output":          {"type": "text"},
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
            dedupe_strategy: Literal["exact", "similar"],
            labels: list[str],
            output: str,
            prompt_summary: str,
            prompt_summary_emb: list[float],
            search_keywords: list[str],
            tool_arguments: str,
            tool_name: str,
            topics: list[str],
    ):
        now = datetime.now(UTC)
        doc = ToolCallDoc(
            doc_id=call_id,
            created_at=now,
            labels=labels,
            output=output,
            search_keywords=search_keywords,
            tool_arguments=tool_arguments,
            tool_name=tool_name,
        )
        try:
            await self.es.index(
                index="tool-calls",
                id=call_id,
                document=(doc.model_dump() | {
                    "dedupe_strategy": dedupe_strategy,
                    "prompt_summary": prompt_summary,
                    "prompt_summary_emb": prompt_summary_emb,
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
            min_initial_score: float = 0.5,
            min_output_similarity_score: float = 0.9,
            not_labels_and: list[str] | None = None,
            not_labels_or: list[str] | None = None,
    ) -> list[ToolCallDoc]:
        del min_output_similarity_score  # Currently unused: result dedupe by output similarity is disabled.
        shared_filters = []
        if labels_and:
            for labels in labels_and:
                shared_filters.append({"term": {"labels": labels}})
        if labels_or:
            shared_filters.append({"terms": {"labels": labels_or}})

        shared_must_not = [{"term": {"labels": "deactivated"}}]
        if excluded_ids:
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
            match_tokens = [
                token
                for token in (strip_punctuation(tok) for tok in match.split())
                if token
            ]
            should_clauses = [
                {"match": {"output": {"query": match, "boost": 1.0}}},
                {"match": {"prompt_summary": {"query": match, "boost": 1.0}}},
                #{"match": {"tool_arguments": {"query": match, "boost": 1.0}}},
                #{"match_phrase": {"tool_arguments": {"query": match, "boost": 1.0}}},
            ]
            if match_tokens:
                should_clauses.append({"terms": {"search_keywords": match_tokens, "boost": 5.0}})
            if similar_topics:
                should_clauses.append({"terms": {"topics": similar_topics, "boost": 1.0}})
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

        knn_hits = []
        if match_emb:
            knn_search_kwargs = {
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
            knn_response = await self.es.search(**knn_search_kwargs)
            knn_hits = knn_response["hits"]["hits"]
            logger.debug(f"Search tool call knn hit count: {len(knn_hits)}")

        hits_by_id = {h["_id"]: h for h in lexical_hits}
        for h in knn_hits:
            if h["_id"] not in hits_by_id:
                hits_by_id[h["_id"]] = h
        hits = list(hits_by_id.values())
        logger.debug(f"Search tool call merged hit count: {len(hits)}")

        lexical_scores = {h["_id"]: h.get("_score", 0.0) for h in lexical_hits}
        knn_scores = {h["_id"]: h.get("_score", 0.0) for h in knn_hits}

        lexical_max = max(lexical_scores.values(), default=0.0)
        semantic_max = max(knn_scores.values(), default=0.0)
        lexical_scores_norm = {
            doc_id: (score / lexical_max) if lexical_max else 0.0
            for doc_id, score in lexical_scores.items()
        }
        semantic_scores_norm = {
            doc_id: (score / semantic_max) if semantic_max else 0.0
            for doc_id, score in knn_scores.items()
        }

        # Penalty-only use of semantics: dissimilar candidates are pushed down.
        dissimilarity_floor = 0.6
        dissimilarity_penalty_weight = 0.5

        deduped_hits = {}
        for h in hits:
            tool_arguments = h["_source"].get("tool_arguments")
            tool_name = h["_source"].get("tool_name")
            lexical_norm = lexical_scores_norm.get(h["_id"], 0.0)
            semantic_norm = semantic_scores_norm.get(h["_id"], 0.0)
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
                f" | lexical: {lexical_norm:.4f}"
                f" | semantic: {semantic_norm:.4f}"
                f" | penalty: {dissimilarity_penalty:.4f}"
                f" | final: {final_score:.4f}"
                f" | Tool: {tool_name} | Args: {tool_arguments}"
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
        results = [ToolCallDoc(doc_id=h["_id"], **h["_source"]) for h in sorted_hits[:limit]]
        logger.debug(f"Search tool call final hit count: {len(results)}")
        return results

    async def search_tool_call_by_arguments_hash(
        self,
        tool_name: str,
        tool_arguments_hash: str,
        size: int = None,
    ) -> list[ToolCallDoc]:
        search_kwargs = {
            "index": "tool-calls",
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"tool_name": tool_name}},
                        {"term": {"tool_arguments_hash": tool_arguments_hash}},
                    ],
                    "must_not": [
                        {"term": {"labels": "deactivated"}},
                    ],
                }
            },
            "sort": [{"created_at": {"order": "desc"}}],
        }
        if size:
            search_kwargs["size"] = size
        response = await self.es.search(**search_kwargs)
        hits = response["hits"]["hits"]
        return [ToolCallDoc(doc_id=h["_id"], **h["_source"]) for h in hits]

    async def search_tool_call_by_arguments_similarity(
        self,
        tool_name: str,
        tool_arguments: str,
        excluded_arguments_hashes: list[str],
        min_score: float = 0.5,
        size: int = 5,
    ) -> list[ToolCallDoc]:
        query = {
            "bool": {
                "filter": [
                    {"term": {"tool_name": tool_name}},
                ],
                "must_not": [
                    {"term": {"labels": "deactivated"}},
                    *[
                        {"term": {"tool_arguments_hash": arguments_hash}}
                        for arguments_hash in excluded_arguments_hashes
                    ],
                ],
                "must": [
                    # Near-exact argument match
                    {"match_phrase": {"tool_arguments": {"query": tool_arguments}}},
                ],
            }
        }
        response = await self.es.search(
            index="tool-calls",
            query=query,
            min_score=min_score,
            size=size,
        )
        hits = response["hits"]["hits"]
        return [ToolCallDoc(doc_id=h["_id"], **h["_source"]) for h in hits]
