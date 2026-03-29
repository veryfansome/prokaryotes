import hashlib
import logging
from abc import (
    ABC,
    abstractmethod,
)

from elasticsearch import AsyncElasticsearch, helpers

logger = logging.getLogger(__name__)

topic_mappings = {
    "dynamic": "strict",
    "properties": {
        "emb": {
            "type":       "dense_vector",
            "dims":       256,
            "index":      True,
            "similarity": "cosine",
        },
        "name": {
            "type": "text",
            "fields": {
                "keyword": {
                    "type": "keyword",
                    "ignore_above": 256,
                }
            }
        },
    }
}


class TopicSearcher(ABC):
    @property
    @abstractmethod
    def es(self) -> AsyncElasticsearch:
        pass

    async def index_topics(self, topics: list[str], topic_embs: list[list[float]]):
        # TODO: Move hashing further upstream and do ID lookup to skip embedding generation
        actions = [
            {
                "_index": "topics",
                "_id": generate_topic_id(topic),
                "_op_type": "create",
                "_source": {"emb": topic_embs[idx], "name": topic}
            }
            for idx, topic in enumerate(topics)
        ]
        success_cnt, errors = await helpers.async_bulk(self.es, actions, raise_on_error=False)
        if success_cnt:
            logger.info(f"Indexed {success_cnt} topic(s)")
        if errors:
            skipped_cnt = len([e for e in errors if e.get("create", {}).get("status") == 409])  # type: ignore 
            if skipped_cnt:
                logger.error(f"Skipped indexing {skipped_cnt} topic(s)")
            error_cnt = len(errors) - skipped_cnt
            if error_cnt:
                logger.error(f"Failed to index {error_cnt} topic(s)")

    async def search_topics(
            self,
            match: str,
            match_emb: list[float],
            knn_num_candidates: int = 100,
            knn_top_k: int = 10,
            min_score: float = 0.5,
    ) -> list[str]:
        query = {
            "should": [
                {"match": {"name": {"query": match, "boost": 1.0}}},
                {"term": {"name.keyword": {"value": match, "boost": 5.0}}}
            ]
        }
        search_kwargs = {
            "index": "topics",
            "query": {
                "bool": query,
            },
        }
        if match_emb:
            search_kwargs["knn"] = {
                "field": "emb",
                "query_vector": match_emb,
                "boost": 1.0,
                "num_candidates": knn_num_candidates,
                "k": knn_top_k,
            }
        response = await self.es.search(**search_kwargs)
        hits = response["hits"]["hits"]
        for h in hits:
            name = h['_source']['name']
            logger.debug(f"Score: {h['_score']:.4f} | Topic: {name}")
        # TODO: Not efficient or scalable, pass min_score to es.search()
        return [h["_source"]["name"] for h in hits if not min_score or (h["_score"] >= min_score)]


def generate_topic_id(text: str) -> str:
    """Generate a consistent, URL-safe ID"""
    return hashlib.md5(text.lower().strip().encode("utf-8")).hexdigest()
