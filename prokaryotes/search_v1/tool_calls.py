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
        if match:
            match_tokens = {strip_punctuation(tok) for tok in match.split()}
            main_query["should"] = [
                {"match": {"output": {"query": match, "boost": 1.0}}},
                {"match": {"prompt_summary": {"query": match, "boost": 1.0}}},
                #{"match": {"tool_arguments": {"query": match, "boost": 1.0}}},
                #{"match_phrase": {"tool_arguments": {"query": match, "boost": 1.0}}},
                {"terms": {"search_keywords": list(match_tokens), "boost": 5.0}},
                {"terms": {"topics": similar_topics, "boost": 1.0}},
            ]
        search_kwargs = {
            "index": "tool-calls",
            "query": {
                "bool": main_query,
            },
            "size": candidates,
        }
        if min_initial_score:
            search_kwargs["min_score"] = min_initial_score
        if match_emb:
            search_kwargs["knn"] = {
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
        response = await self.es.search( **search_kwargs)
        hits = response["hits"]["hits"]
        logger.debug(f"Search tool call initial hit count: {len(hits)}")

        deduped_hits = {}
        for h in hits:
            tool_arguments = h["_source"].get("tool_arguments")
            tool_name = h["_source"].get("tool_name")
            logger.debug(
                f"ToolCallDoc ID: {h['_id']} | Score: {h['_score']:.4f}"
                f" | Tool: {tool_name} | Args: {tool_arguments}"
            )
            tool_sig = (tool_name, tool_arguments)
            if tool_sig in deduped_hits:
                if deduped_hits[tool_sig]["_source"]["created_at"] < h["_source"].get("created_at"):
                    deduped_hits[tool_sig] = h
                continue
            deduped_hits[tool_sig] = h
        logger.debug(f"Search tool call post-dedupe hit count: {len(deduped_hits)}")

        # TODO: deduped_hits needs to be first deduped on similarity
        # TODO: We should also do similarity dedupe against what's already in the context window

        results = []
        #arguments_hashes = await run_in_threadpool(
        #    text_to_md5_batch,
        #    [tool_arguments for (_, tool_arguments) in deduped_hits.keys()]
        #)
        #for idx, ((tool_name, tool_arguments), h) in enumerate(deduped_hits.items()):
        #    if h["_source"]["dedupe_strategy"] == "exact":
        #        results.extend(await self.search_tool_call_by_arguments_hash(
        #            tool_name, arguments_hashes[idx], size=1
        #        ))
        #    else:
        #        h_doc = ToolCallDoc(doc_id=h["_id"], **h["_source"])
        #        similar_calls = await self.search_tool_call_by_arguments_similarity(
        #            tool_name, tool_arguments, arguments_hashes
        #        )
        #        if similar_calls:
        #            output_scores = await run_in_threadpool(
        #                str_similarity_batch,
        #                h_doc.output,
        #                [call.output for call in similar_calls]
        #            )
        #            if all(score < min_output_similarity_score for score in output_scores):
        #                results.append(h_doc)
        #            else:
        #                sorted_calls = sorted(
        #                    [(h_doc.created_at, h_doc, 1.0)] + [
        #                        (call.created_at, call, score)
        #                        for call, score in zip(similar_calls, output_scores, strict=True)
        #                        if score >= min_output_similarity_score
        #                    ],
        #                    key=(lambda x: x[0]),  # Sort by call.created_at
        #                    reverse=True
        #                )
        #                results.append(sorted_calls[0][1])
        #        else:
        #            results.append(h_doc)
        #logger.debug(f"Search tool call final hit count: {len(results)}")
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
