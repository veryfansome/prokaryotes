import asyncio
import logging
from abc import (
    ABC,
    abstractmethod,
)
from datetime import datetime, timezone
from elastic_transport import ObjectApiResponse
from elasticsearch import AsyncElasticsearch

from prokaryotes.models_v1 import FactDoc

logger = logging.getLogger(__name__)

fact_mappings = {
    "dynamic": "strict",
    "properties": {
        "about":         {"type": "keyword"},
        "created_at":    {"type": "date"},
        "importance":    {"type": "integer"},
        "invalid_after": {"type": "date"},
        "labels":        {"type": "keyword"},
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


class FactSearcher(ABC):
    @property
    @abstractmethod
    def es() -> AsyncElasticsearch:
        pass

    async def index_facts(self, about: list[str], fact_texts: list[str], fact_embs: list[list[float]]):
        """Index a small list of facts."""
        now = datetime.now(timezone.utc)
        docs = [FactDoc(about=about, created_at=now, text=text) for text in fact_texts]
        index_tasks = [
            self.es.index(index="facts", document=(doc.model_dump() | {"text_emb": fact_embs[idx]}))
            for idx, doc in enumerate(docs)
        ]
        results: list[ObjectApiResponse | Exception] = await asyncio.gather(*index_tasks, return_exceptions=True)
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to index {docs[idx]}", exc_info=result)
                # TODO: Retry?
            else:
                docs[idx].doc_id = result["_id"]
        return docs

    async def knn_dedupe_facts(
            self,
            about: str,
            match_emb: list[float],
            score_threshold: float = 0.4,
    ):
        now = datetime.now(tz=timezone.utc)
        response = await self.es.search(
            index="facts",
            knn={
                "field": "text_emb",
                "query_vector": match_emb,
                "k": 50,
                "num_candidates": 100,
                "filter": {
                    "bool": {
                        "must": [
                            {"term": {"about": about}},
                            {
                                "bool": {
                                    "should": [
                                        {"bool": {"must_not": {"exists": {"field": "invalid_after"}}}},
                                        {"range": {"invalid_after": {"gt": now.isoformat()}}},
                                    ],
                                    "minimum_should_match": 1,
                                }
                            }
                        ],
                        "must_not": [{"term": {"labels": "deactivated"}}]
                    }
                }
            }
        )
        hits = response["hits"]["hits"]
        for h in hits:
            text = h['_source'].get('text')
            displayed_text = text if len(text) <= 50 else (text[:50] + "...")
            logger.debug(f"FactDoc ID: {h['_id']} | Score: {h['_score']:.4f} | Text: {displayed_text}")
        return [FactDoc(doc_id=h["_id"], **h["_source"]) for h in hits if h["_score"] >= score_threshold]

    async def search_facts(
            self,
            about: str | list[str] | None = None,
            knn_num_candidates: int = 100,
            knn_top_k: int = 50,
            match: str = None,
            match_emb: list[float] = None,
            min_score: float = None,
    ) -> list[FactDoc]:
        now = datetime.now(tz=timezone.utc)
        shared_filters = []
        if about:
            if isinstance(about, str):
                shared_filters.append({"term": {"about": about}})
            else:
                shared_filters.append({"terms": {"about": about}}) # OR
        shared_filters.append({
            "bool": {
                "should": [
                    {"bool": {"must_not": {"exists": {"field": "invalid_after"}}}},
                    {"range": {"invalid_after": {"gt": now.isoformat()}}},
                ],
                "minimum_should_match": 1,
            }
        })
        main_query = {
            "filter": shared_filters,
            "must_not": [{"term": {"labels": "deactivated"}}]
        }
        if match:
            main_query["should"] = [{"match": {"text": match}}]
        search_kwargs = {
            "index": "facts",
            "query": {
                "function_score": {
                    "query": {"bool": main_query},
                    "functions": [
                        # Gaussian Recency Decay
                        {
                            "gauss": {
                                "created_at": {
                                    "origin": "now",
                                    "scale": "7d",  # Score drops to 'decay' value at this age
                                    "offset": "1h",  # Documents newer than 1h get full score
                                    "decay": 0.5  # At 7 days old
                                }
                            }
                        },
                        # Importance Multiplier
                        {
                            "field_value_factor": {
                                "field": "importance",
                                "factor": 1.0,
                                "missing": 1  # Default if empty
                            }
                        }
                    ],
                    "score_mode": "multiply",
                    "boost_mode": "multiply",
                },
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
                        "must": shared_filters,
                        "must_not": [{"term": {"labels": "deactivated"}}]
                    }
                }
            }
        response = await self.es.search(**search_kwargs)
        hits = response["hits"]["hits"]
        for h in hits:
            text = h['_source'].get('text')
            displayed_text = text if len(text) <= 50 else (text[:50] + "...")
            logger.debug(f"FactDoc ID: {h['_id']} | Score: {h['_score']:.4f} | Text: {displayed_text}")
        return [FactDoc(doc_id=h["_id"], **h["_source"])
                for h in hits if not min_score or (h["_score"] >= min_score)]
