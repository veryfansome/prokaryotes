import logging
import os
from datetime import datetime, timezone
from elasticsearch import AsyncElasticsearch
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class PersonFactDoc(BaseModel):
    created_at: datetime
    importance: int = 0
    invalid_after: datetime | None = None
    labels: list[str] = Field(default_factory=list)
    text: str

class PersonQuestionDoc(BaseModel):
    created_at: datetime
    importance: int = 0
    invalid_after: datetime | None = None
    labels: list[str] = Field(default_factory=list)
    text: str

class PersonDoc(BaseModel):
    created_at: datetime
    doc_id: str | None = Field(default=None, exclude=True)
    facts: list[PersonFactDoc] = Field(default_factory=list)
    questions: list[PersonQuestionDoc] = Field(default_factory=list)
    user_id: str

person_mappings = {
    "dynamic": "strict",
    "properties": {
        "created_at": {"type": "date"},
        "facts": {
            "type": "nested",
            "properties": {
                "created_at":    {"type": "date"},
                "importance":    {"type": "integer"},
                "invalid_after": {"type": "date"},
                "labels":        {"type": "keyword"},
                "text":          {"type": "text"},
            }
        },
        "labels": {"type": "keyword"},
        "questions": {
            "type": "nested",
            "properties": {
                "created_at":    {"type": "date"},
                "importance":    {"type": "integer"},
                "invalid_after": {"type": "date"},
                "labels":        {"type": "keyword"},
                "text":          {"type": "text"},
            }
        },
        "user_id": {"type": "keyword"},
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

    async def add_user_person_fact_doc(self, person_doc: PersonDoc, fact_texts: list[str]):
        created_at = datetime.now(timezone.utc)
        fact_docs = [PersonFactDoc(created_at=created_at, text=text) for text in fact_texts]
        await self.es.update(
            index="persons",
            id=person_doc.doc_id,
            script={
                "source": """
                    if (ctx._source.facts == null) { 
                        ctx._source.facts = [] 
                    } 
                    ctx._source.facts.addAll(params.new_facts)
                """,
                "params": {
                    "new_facts": [doc.model_dump() for doc in fact_docs]
                }
            }
        )
        person_doc.facts.extend(fact_docs)

    async def add_user_person_question_doc(self, person_doc: PersonDoc, question_texts: list[str]):
        created_at = datetime.now(timezone.utc)
        question_docs = [PersonQuestionDoc(created_at=created_at, text=text) for text in question_texts]
        await self.es.update(
            index="persons",
            id=person_doc.doc_id,
            script={
                "source": """
                    if (ctx._source.questions == null) { 
                        ctx._source.questions = [] 
                    } 
                    ctx._source.questions.addAll(params.new_questions)
                """,
                "params": {
                    "new_questions": [doc.model_dump() for doc in question_docs]
                }
            }
        )
        person_doc.questions.extend(question_docs)

    async def get_or_create_user_person_doc(self, user_id: str, max_inner_hits: int = 100) -> PersonDoc:
        now = datetime.now(tz=timezone.utc).isoformat()
        response = await self.es.search(
            index="persons",
            _source={"excludes": ["facts", "questions"]},
            query={
                "bool": {
                    "filter":   [{"term": {"user_id": user_id}}],
                    "must_not": [{"term": {"labels": "deactivated"}}],
                    "should": [
                        {
                            "nested": {
                                "path": "facts",
                                "query": {
                                    "bool": {
                                        "should": [
                                            {"bool": {"must_not": {"exists": {"field": "facts.invalid_after"}}}},
                                            {"range": {"facts.invalid_after": {"gt": now}}},
                                        ]
                                    }
                                },
                                "inner_hits": {
                                    "name": "active_facts",
                                    "size": max_inner_hits,
                                }
                            }
                        },
                        {
                            "nested": {
                                "path": "questions",
                                "query": {
                                    "bool": {
                                        "filter": [
                                            # 1. Must not be deactivated (AND)
                                            {"bool": {"must_not": {"term": {"questions.labels": "deactivated"}}}}
                                        ],
                                        "should": [
                                            # 2. Must meet one of these date conditions (OR)
                                            {"bool": {"must_not": {"exists": {"field": "questions.invalid_after"}}}},
                                            {"range": {"questions.invalid_after": {"gt": now}}}
                                        ],
                                        "minimum_should_match": 1,
                                    }
                                },
                                "inner_hits": {
                                    "name": "active_questions",
                                    "size": max_inner_hits,
                                }
                            }
                        },
                    ],
                }
            },
            sort=[{"created_at": {"order": "desc"}}],  # Prefers recency in case of duplicates
        )
        hits = response["hits"]["hits"]
        if hits:
            doc_source = hits[0]["_source"]
            inner_hits_data = hits[0].get("inner_hits", {})
            inner_facts = [
                h["_source"] for h in inner_hits_data.get("active_facts", {}).get("hits", {}).get("hits", [])
            ]
            inner_questions = [
                h["_source"] for h in inner_hits_data.get("active_questions", {}).get("hits", {}).get("hits", [])
            ]
            return PersonDoc(doc_id=hits[0]["_id"], facts=inner_facts, questions=inner_questions, **doc_source)
        else:
            created_at = datetime.now(timezone.utc)
            response = await self.es.index(index="persons", document={
                "created_at": created_at.isoformat(), "facts": [], "questions": [], "user_id": user_id,
            })
            return PersonDoc(created_at=created_at, doc_id=response["_id"], user_id=user_id)

    async def close(self):
        await self.es.close()