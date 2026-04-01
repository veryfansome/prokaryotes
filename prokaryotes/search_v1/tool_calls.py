import logging
from abc import (
    ABC,
    abstractmethod,
)
from datetime import UTC, datetime

from elasticsearch import AsyncElasticsearch

from prokaryotes.models_v1 import ToolCallDoc

logger = logging.getLogger(__name__)

tool_call_mappings = {
    "dynamic": "strict",
    "properties": {
        "created_at": {"type": "date"},
        "labels":     {"type": "keyword"},
        "output":     {"type": "text"},
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
        "tool_arguments": {"type": "text"},
        "tool_name":      {"type": "keyword"},
    }
}


class ToolCallSearcher(ABC):
    @property
    @abstractmethod
    def es(self) -> AsyncElasticsearch:
        pass

    async def index_tool_call(
            self,
            labels: list[str],
            output: str,
            prompt_summary: str,
            prompt_summary_emb: list[float],
            tool_name: str,
            tool_arguments: str,
    ):
        now = datetime.now(UTC)
        doc = ToolCallDoc(
            created_at=now,
            labels=labels,
            output=output,
            prompt_summary=prompt_summary,
            tool_arguments=tool_arguments,
            tool_name=tool_name,
        )
        try:
            result = await self.es.index(
                index="tool-calls",
                document=(doc.model_dump() | {"prompt_summary_emb": prompt_summary_emb})
            )
            doc.doc_id = result["_id"]
            return doc
        except Exception:
            logger.exception(f"Failed to index {doc}")

    async def search_tool_call(
            self,
            knn_num_candidates: int = 100,
            knn_top_k: int = 50,
            labels_and: list[str] | None = None,
            labels_or: list[str] | None = None,
            match: str = None,
            match_emb: list[float] = None,
            min_score: float = 0.5,
            not_labels_and: list[str] | None = None,
            not_labels_or: list[str] | None = None,
            size: int = 3,
    ) -> list[ToolCallDoc]:
        shared_filters = []
        if labels_and:
            for labels in labels_and:
                shared_filters.append({"term": {"labels": labels}})
        if labels_or:
            shared_filters.append({"terms": {"labels": labels_or}})

        shared_must_not = [{"term": {"labels": "deactivated"}}]
        if not_labels_and:
            for labels in not_labels_and:
                shared_must_not.append({"term": {"labels": labels}})
        if not_labels_or:
            shared_must_not.append({"terms": {"labels": not_labels_or}})

        main_query = {
            "filter": shared_filters,
            "must_not": shared_must_not,
        }
        if match:
            main_query["should"] = [
                {"match": {"output": {"query": match, "boost": 1.0}}},
                {"match": {"prompt_summary": {"query": match, "boost": 2.0}}},
            ]
        search_kwargs = {
            "index": "tool-calls",
            "query": {
                "bool": main_query,
            },
            "size": size,
        }
        if match_emb:
            search_kwargs["knn"] = {
                "field": "prompt_summary_emb",
                "query_vector": match_emb,
                "boost": 1.0,
                "num_candidates": knn_num_candidates,
                "k": knn_top_k,
                "filter": {
                    "bool": {
                        "filter": shared_filters,
                        "must_not": shared_must_not,
                    }
                }
            }
        response = await self.es.search( **search_kwargs)
        hits = response["hits"]["hits"]
        for h in hits:
            tool_arguments = h['_source'].get('tool_arguments')
            tool_name = h['_source'].get('tool_name')
            logger.debug(
                f"ToolCallDoc ID: {h['_id']} | Score: {h['_score']:.4f}"
                f" | Tool: {tool_name} | Args: {tool_arguments}"
            )
        # TODO: Not efficient or scalable, pass min_score to es.search()
        return [ToolCallDoc(doc_id=h["_id"], **h["_source"])
                for h in hits if not min_score or h["_score"] >= min_score]
