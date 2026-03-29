import asyncio
import logging
from abc import (
    ABC,
    abstractmethod,
)
from elastic_transport import ObjectApiResponse
from elasticsearch import AsyncElasticsearch

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
    def es() -> AsyncElasticsearch:
        pass

    async def index_topics(self, topics: list[str], topic_embs: list[list[float]]):
        index_tasks = [
            self.es.index(index="topics", document={"emb": topic_embs[idx], "name": topic})
            for idx, topic in enumerate(topics)
        ]
        results: list[ObjectApiResponse | Exception] = await asyncio.gather(*index_tasks, return_exceptions=True)
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to index topic: {topics[idx]}", exc_info=result)
