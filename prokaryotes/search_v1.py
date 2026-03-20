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
    ToolCallDoc,
)

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
            "type":            "text",
            "analyzer":        "standard",
            "search_analyzer": "custom_query_analyzer",
        },
    }
}

tool_call_mappings = {
    "dynamic": "strict",
    "properties": {
        "labels": {"type": "keyword"},
        "output": {"type": "object", "enabled": False},
        "anchors": {
            "type": "nested",
            "properties": {
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
            },
        },
        "updated_at": {"type": "date"},
    }
}

class SearchClient:
    def __init__(self):
        self.es: AsyncElasticsearch | None = None

    async def close(self):
        await self.es.close()

    async def get_user_context(
            self,
            full_name: str,
            user_id: int,
            match: str = None,
            match_emb: list[float] = None,
            min_facts_score: float = None,
    ) -> PersonContext:
        facts = await self.search_facts(
            f"user:{user_id}",
            match=match,
            match_emb=match_emb,
            min_score=min_facts_score,
        )
        return PersonContext(facts=facts, name=full_name, questions=[], user_id=user_id)

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

    async def index_prompt(
            self,
            about: list[str],
            prompt_uuid: str,
            labels: list[str],
            messages: list[ChatMessage],
    ) -> PromptDoc | None:
        now = datetime.now(timezone.utc)
        doc = PromptDoc(
            about=about, created_at=now, doc_id=prompt_uuid, labels=labels, messages=messages
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
        now = datetime.now(timezone.utc)
        doc = ResponseDoc(
            about=about, created_at=now, doc_id=prompt_uuid, error=error, labels=labels, text=generated_response
        )
        try:
            await self.es.index(id=prompt_uuid, index="responses", document=doc.model_dump())
            return doc
        except Exception:
            logger.exception(f"Failed to index {doc}")

    async def index_tool_call(
            self,
            labels: list[str],
            output: str,
            anchor_emb: list[float] = None,
            anchor_text: str = None,
    ):
        now = datetime.now(timezone.utc)
        doc = ToolCallDoc(
            anchors=[], labels=labels, output=output, updated_at=now,
        )
        anchor = {}
        if anchor_text:
            anchor["text"] = anchor_text
            if anchor_emb:
                anchor["text_emb"] = anchor_emb
        try:
            result = await self.es.index(
                index="tool-calls",
                document=(doc.model_dump() | {"anchors": [anchor]} if anchor_text else {})
            )
            doc.doc_id = result["_id"]
            return doc
        except Exception:
            logger.exception(f"Failed to index {doc}")

    def init_client(self):
        self.es = get_elastic_search()

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
            about: str,
            knn_num_candidates: int = 100,
            knn_top_k: int = 50,
            match: str = None,
            match_emb: list[float] = None,
            min_score: float = None,
    ) -> list[FactDoc]:
        now = datetime.now(tz=timezone.utc)
        shared_filters = [
            # TODO: Might need to expand to searching multiple about keywords with AND/OR
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

    async def search_tool_call_by_labels(self, filter_labels: list[str]) -> list[ToolCallDoc]:
        response = await self.es.search(
            index="tool-calls",
            query={
                "bool": {
                    "filter": [{"term": {"labels": label}} for label in filter_labels]
                }
            }
        )
        hits = response["hits"]["hits"]
        for h in hits:
            labels = h['_source'].get('labels')
            logger.debug(f"ToolCallDoc ID: {h['_id']} | Labels: {labels}")
        return [ToolCallDoc(doc_id=h["_id"], **h["_source"]) for h in hits]

    async def search_tool_call_by_anchor(
            self,
            match: str,
            match_emb: list[float],
            min_score: float = 1.5,
            knn_num_candidates: int = 100,
            knn_top_k: int = 30,
            top_k: int = 10,
    ) -> list[ToolCallDoc]:
        response = await self.es.search(
            index="tool-calls",
            knn={
                "field": "anchors.text_emb",
                "query_vector": match_emb,
                "boost": 1.0,
                "k": knn_top_k,
                "num_candidates": knn_num_candidates,
            },
            query={
                "bool": {
                    "should": [
                        {
                            "nested": {
                                "path": "anchors",
                                "query": {
                                    "match": {
                                        "anchors.text": {
                                            "query": match,
                                            "boost": 1.0
                                        }
                                    }
                                },
                            }
                        }
                    ]
                }
            },
            size=top_k,
        )
        hits = response["hits"]["hits"]
        for h in hits:
            labels = h['_source'].get('labels')
            logger.debug(f"ToolCallDoc ID: {h['_id']} | Score: {h['_score']:.4f} | Labels: {labels}")
        return [ToolCallDoc(doc_id=h["_id"], **h["_source"]) for h in hits if h["_score"] >= min_score]

    async def update_tool_call(
            self,
            doc_id: str,
            anchor_emb: list[float] = None,
            anchor_text: str = None,
            output: str = None,
    ):
        now = datetime.now(timezone.utc)
        script = [
            "ctx._source.updated_at = params.updated_at;",
        ]
        anchor = {}
        if anchor_text:
            script.append("ctx._source.anchors.add(params.anchor);")
            anchor["text"] = anchor_text
            if anchor_emb:
                anchor["text_emb"] = anchor_emb
        if output:
            script.append("ctx._source.output = params.output;")
        try:
            await self.es.update(
                index="tool-calls",
                id=doc_id,
                body={
                    "script": {
                        "source": "\n".join(script),
                        "params": (
                            {"updated_at": now.isoformat()}
                            | ({"anchor": anchor} if anchor_text else {})
                            | ({"output": output} if output else {})
                        )
                    }
                }
            )
        except Exception:
            logger.exception(f"Failed to update tool-calls document: id={doc_id}")

def get_elastic_search() -> AsyncElasticsearch:
    uri = os.environ.get("ELASTIC_URI")
    if uri:
        return AsyncElasticsearch(uri)
    raise RuntimeError("Unable to initialize Elasticsearch client")
