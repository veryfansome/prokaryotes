import asyncio
import logging
import os
from datetime import datetime, timezone
from elastic_transport import ObjectApiResponse
from elasticsearch import AsyncElasticsearch

from prokaryotes.models_v1 import (
    FactDoc,
    PersonContext,
    QuestionDoc,
)

logger = logging.getLogger(__name__)

fact_mappings = {
    "dynamic": "strict",
    "properties": {
        "about": {"type": "keyword"},
        "created_at": {"type": "date"},
        "importance": {"type": "integer"},
        "invalid_after": {"type": "date"},
        "labels": {"type": "keyword"},
        "text": {"type": "text"},
    }
}

question_mappings = {
    "dynamic": "strict",
    "properties": {
        "about": {"type": "keyword"},
        "created_at": {"type": "date"},
        "importance": {"type": "integer"},
        "invalid_after": {"type": "date"},
        "labels": {"type": "keyword"},
        "text": {"type": "text"},
        "to": {"type": "keyword"},
    }
}

def get_elastic_search() -> AsyncElasticsearch:
    elastic_uri = os.environ.get("ELASTIC_URI")
    if elastic_uri:
        return AsyncElasticsearch(elastic_uri)
    raise RuntimeError("Unable to initialize Elasticsearch client")

class SearchClient:
    def __init__(self, es: AsyncElasticsearch = get_elastic_search()):
        self.es = es

    async def index_facts(self, about: list[str], fact_texts: list[str]) -> list[FactDoc]:
        """Index a small list of facts."""
        created_at = datetime.now(timezone.utc)
        facts = [FactDoc(about=about, created_at=created_at, text=text) for text in fact_texts]
        index_tasks = [self.es.index(index="facts", document=fact.model_dump()) for fact in facts]
        results: list[ObjectApiResponse | Exception] = await asyncio.gather(*index_tasks, return_exceptions=True)
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to index {facts[idx]}", exc_info=result)
            else:
                facts[idx].doc_id = result["_id"]
        return facts

    async def search_facts(self, about: str, match: str = None) -> list[FactDoc]:
        now = datetime.now(tz=timezone.utc)
        main_query = {
            "filter": [
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
            "must_not": [
                {"term": {"labels": "deactivated"}},
            ]
        }
        if match:
            main_query["should"] = [{"match": {"text": match}}]
        response = await self.es.search(
            index="facts",
            query={
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
        )
        hits = response["hits"]["hits"]
        return [FactDoc(doc_id=h["_id"], **h["_source"]) for h in hits]

    async def get_user_context(self, user_id: int) -> PersonContext:
        facts = await self.search_facts(f"user_{user_id}")
        return PersonContext(facts=facts, questions=[], user_id=user_id)

    async def close(self):
        await self.es.close()