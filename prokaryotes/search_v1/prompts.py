import logging
from abc import (
    ABC,
    abstractmethod,
)
from datetime import UTC, datetime

from elasticsearch import AsyncElasticsearch

from prokaryotes.models_v1 import (
    ChatMessage,
    PromptDoc,
)

logger = logging.getLogger(__name__)

prompt_mappings = {
    "dynamic": "strict",
    "properties": {
        "about":      {"type": "keyword"},
        "created_at": {"type": "date"},
        "labels":     {"type": "keyword"},
        "messages": {
            "type": "nested",
            "properties": {
                "role": {"type": "keyword"},
                "content":  {
                    "type":            "text",
                    "analyzer":        "standard",
                    "search_analyzer": "custom_query_analyzer",
                },
            }
        }
    }
}


class PromptSearcher(ABC):
    @property
    @abstractmethod
    def es(self) -> AsyncElasticsearch:
        pass

    async def index_prompt(
            self,
            about: list[str],
            prompt_uuid: str,
            labels: list[str],
            messages: list[ChatMessage],
    ) -> PromptDoc | None:
        now = datetime.now(UTC)
        doc = PromptDoc(
            about=about, created_at=now, doc_id=prompt_uuid, labels=labels, messages=messages
        )
        try:
            await self.es.index(
                id=prompt_uuid,
                index="prompts",
                document=doc.model_dump(exclude={"messages": {"__all__": "type"}})
            )
            return doc
        except Exception:
            logger.exception(f"Failed to index {doc}")
