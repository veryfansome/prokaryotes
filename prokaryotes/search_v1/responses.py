import logging
from abc import (
    ABC,
    abstractmethod,
)
from datetime import UTC, datetime

from elasticsearch import AsyncElasticsearch

from prokaryotes.models_v1 import ResponseDoc

logger = logging.getLogger(__name__)

response_mappings = {
    "dynamic": "strict",
    "properties": {
        "about":      {"type": "keyword"},
        "created_at": {"type": "date"},
        "labels":     {"type": "keyword"},
        "text": {
            "type":            "text",
            "analyzer":        "standard",
            "search_analyzer": "custom_query_analyzer",
        },
        "text_emb": {
            "type":       "dense_vector",
            "dims":       256,
            "index":      True,
            "similarity": "cosine",
        },
    }
}


class ResponseSearcher(ABC):
    @property
    @abstractmethod
    def es(self) -> AsyncElasticsearch:
        pass

    async def index_response(
            self,
            about: list[str],
            labels: list[str],
            prompt_uuid: str,
            response_emb: list[float],
            response_text: str,
    ):
        now = datetime.now(UTC)
        doc = ResponseDoc(about=about, created_at=now, doc_id=prompt_uuid, labels=labels, text=response_text)
        try:
            await self.es.index(
                id=prompt_uuid, index="responses", document=(doc.model_dump() | {"text_emb": response_emb})
            )
            return doc
        except Exception:
            logger.exception(f"Failed to index {doc}")

    async def search_responses(
            self,
            about_and: list[str] | None = None,
            about_or: list[str] | None = None,
            knn_num_candidates: int = 100,
            knn_top_k: int = 50,
            labels_and: list[str] | None = None,
            labels_or: list[str] | None = None,
            match: str = None,
            match_emb: list[float] = None,
            min_score: float = 0.5,
            not_about: str | list[str] | None = None,
    ) -> list[ResponseDoc]:
        shared_filters = []
        if labels_and:
            for labels in labels_and:
                shared_filters.append({"term": {"labels": labels}})
        if labels_or:
            shared_filters.append({"terms": {"labels": labels_or}})

        shared_must_not = [{"term": {"labels": "deactivated"}}]
        if not_about:
            if isinstance(not_about, str):
                shared_must_not.append({"term": {"about": not_about}})
            else:
                shared_must_not.append({"terms": {"about": not_about}})
        main_query = {
            "filter": shared_filters,
            "must_not": shared_must_not,
        }
        if match:
            should = [{"match": {"text": {"query": match, "boost": 1.0}}}]
            if about_and:
                for about in about_and:
                    should.append({"term": {"about": {"value": about, "boost": 2.0}}})
            if about_or:
                should.append({"terms": {"about": about_or, "boost": 2.0}})
            main_query["should"] = should
        search_kwargs = {
            "index": "responses",
            "query": {
                "bool": main_query,
            },
        }
        if match_emb:
            search_kwargs["knn"] = {
                "field": "text_emb",
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
        response = await self.es.search(**search_kwargs)
        hits = response["hits"]["hits"]
        for h in hits:
            text = h['_source'].get('text')
            displayed_text = text if len(text) <= 50 else (text[:50] + "...")
            logger.debug(f"ResponseDoc ID: {h['_id']} | Score: {h['_score']:.4f} | Text: {displayed_text}")
        # TODO: Not efficient or scalable, pass min_score to es.search()
        return [ResponseDoc(doc_id=h["_id"], **h["_source"])
                for h in hits if not min_score or (h["_score"] >= min_score)]
