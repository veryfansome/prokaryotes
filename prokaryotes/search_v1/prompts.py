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
        },
        "named_entities": {"type": "keyword"},
        "topics":         {"type": "keyword"},
    }
}


class PromptSearcher(ABC):
    @property
    @abstractmethod
    def es(self) -> AsyncElasticsearch:
        pass

    async def get_previous_prompt_by_conversation(self, conversation_uuid: str) -> PromptDoc | None:
        conversation_label = f"conversation:{conversation_uuid}"
        try:
            response = await self.es.search(
                index="prompts",
                query={
                    "bool": {
                        "filter": [
                            {"term": {"labels": conversation_label}},
                        ]
                    }
                },
                sort=[{"created_at": {"order": "desc"}}],
                size=1,
            )
            hits = response["hits"]["hits"]
            if not hits:
                return None
            hit = hits[0]
            return PromptDoc(doc_id=hit["_id"], **hit["_source"])
        except Exception:
            logger.exception(f"Failed to retrieve previous prompt for {conversation_label}")
        return None

    async def index_prompt(
            self,
            labels: list[str],
            messages: list[ChatMessage],
            named_entities: list[str],
            prompt_uuid: str,
            topics: list[str],
    ) -> PromptDoc | None:
        now = datetime.now(UTC)
        doc = PromptDoc(
            created_at=now,
            doc_id=prompt_uuid,
            labels=labels,
            messages=messages,
            named_entities=named_entities,
            topics=topics,
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
