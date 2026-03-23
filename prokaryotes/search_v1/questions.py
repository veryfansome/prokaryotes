import logging
from abc import (
    ABC,
    abstractmethod,
)
from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)

question_mappings = {
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
        "to": {"type": "keyword"},
    }
}


class QuestionSearcher(ABC):
    @property
    @abstractmethod
    def es() -> AsyncElasticsearch:
        pass
