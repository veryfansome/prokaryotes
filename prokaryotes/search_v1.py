import asyncio
import logging
import os
from datetime import datetime, timezone
from elastic_transport import ObjectApiResponse
from elasticsearch import AsyncElasticsearch

from prokaryotes.models_v1 import (
    PromptDoc,
    ChatMessage,
    FactDoc,
    PersonContext,
    QuestionDoc,
    ResponseDoc,
    TextEmbeddingPrompt,
    TextEmbeddingRequest,
)
from prokaryotes.utils import get_text_embeddings

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

prompt_mappings = {
    "dynamic": "strict",
    "properties": {
        "about":      {"type": "keyword"},
        "created_at": {"type": "date"},
        "importance": {"type": "integer"},
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

response_mappings = {
    "dynamic": "strict",
    "properties": {
        "about":      {"type": "keyword"},
        "created_at": {"type": "date"},
        "error":      {"type": "text"},
        "importance": {"type": "integer"},
        "labels":     {"type": "keyword"},
        "text": {
            "type": "text",
            "analyzer": "standard",
            "search_analyzer": "custom_query_analyzer",
        },
    }
}

class SearchClient:
    def __init__(self):
        self.es: AsyncElasticsearch | None = None

    async def index_facts(self, about: list[str], fact_texts: list[str]):
        """Index a small list of facts."""
        embs_resp = await get_text_embeddings(
            TextEmbeddingRequest(batch_size=16, prompt=TextEmbeddingPrompt.DOCUMENT, texts=fact_texts, truncate_to=256)
        )
        created_at = datetime.now(timezone.utc)
        facts = [FactDoc(about=about, created_at=created_at, text=text) for text in fact_texts]
        index_tasks = [
            self.es.index(index="facts", document=(fact.model_dump() | {"text_emb": embs_resp.embeddings[idx]}))
            for idx, fact in enumerate(facts)
        ]
        results: list[ObjectApiResponse | Exception] = await asyncio.gather(*index_tasks, return_exceptions=True)
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to index {facts[idx]}", exc_info=result)
                # TODO: Retry?
            else:
                facts[idx].doc_id = result["_id"]
        return facts

    async def index_prompt(
            self,
            about: list[str],
            prompt_uuid: str,
            labels: list[str],
            messages: list[ChatMessage],
    ) -> PromptDoc | None:
        created_at = datetime.now(timezone.utc)
        doc = PromptDoc(
            about=about, created_at=created_at, doc_id=prompt_uuid, labels=labels, messages=messages
        )
        try:
            await self.es.index(id=prompt_uuid, index="prompts", document=doc.model_dump())
            return doc
        except Exception:
            logger.exception(f"Failed to index {doc}")

    async def index_response(
            self,
            about: list[str],
            prompt_uuid: str,
            labels: list[str],
            generated_response: str,
            error: str = None,
    ):
        created_at = datetime.now(timezone.utc)
        doc = ResponseDoc(
            about=about, created_at=created_at, doc_id=prompt_uuid, error=error, labels=labels, text=generated_response
        )
        try:
            await self.es.index(id=prompt_uuid, index="responses", document=doc.model_dump())
            return doc
        except Exception:
            logger.exception(f"Failed to index {doc}")

    def init_client(self):
        self.es = get_elastic_search()

    async def search_facts(self, about: str, match: str = None, match_emb: list[float] = None) -> list[FactDoc]:
        now = datetime.now(tz=timezone.utc)
        shared_filters = [
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
        ]
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
                "boost": 5.0,  # Increase this if semantic matches should weigh more
                "k": 50,
                "num_candidates": 100,
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
            logger.debug(f"Doc ID: {h['_id']} | Score: {h['_score']:.4f} | Text: {displayed_text}...")
        return [FactDoc(doc_id=h["_id"], **h["_source"]) for h in hits]

    async def get_user_context(
            self,
            full_name: str,
            user_id: int,
            match: str = None,
            match_emb: list[float] = None,
    ) -> PersonContext:
        facts = await self.search_facts(f"user_{user_id}", match=match, match_emb=match_emb)
        return PersonContext(facts=facts, name=full_name, questions=[], user_id=user_id)

    async def close(self):
        await self.es.close()

def get_elastic_search() -> AsyncElasticsearch:
    uri = os.environ.get("ELASTIC_URI")
    if uri:
        return AsyncElasticsearch(uri)
    raise RuntimeError("Unable to initialize Elasticsearch client")
