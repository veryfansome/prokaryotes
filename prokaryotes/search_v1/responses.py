import logging
from abc import (
    ABC,
    abstractmethod,
)
from datetime import datetime, timezone
from elasticsearch import AsyncElasticsearch

from prokaryotes.models_v1 import ResponseDoc

logger = logging.getLogger(__name__)

response_mappings = {
    "dynamic": "strict",
    "properties": {
        "about":      {"type": "keyword"},
        "created_at": {"type": "date"},
        "error":      {"type": "text"},
        "importance": {"type": "integer"},
        "labels":     {"type": "keyword"},
        "text": {
            "type":            "text",
            "analyzer":        "standard",
            "search_analyzer": "custom_query_analyzer",
        },
    }
}


class ResponseSearcher(ABC):
    @property
    @abstractmethod
    def es() -> AsyncElasticsearch:
        pass

    async def index_response(
            self,
            about: list[str],
            prompt_uuid: str,
            labels: list[str],
            generated_response: str,
            error: str = None,
    ):
        now = datetime.now(timezone.utc)
        doc = ResponseDoc(
            about=about, created_at=now, doc_id=prompt_uuid, error=error, labels=labels, text=generated_response
        )
        try:
            await self.es.index(id=prompt_uuid, index="responses", document=doc.model_dump())
            return doc
        except Exception:
            logger.exception(f"Failed to index {doc}")
