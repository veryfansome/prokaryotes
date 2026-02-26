import logging
import os
from datetime import datetime, timezone
from elasticsearch import AsyncElasticsearch
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class PersonFactDoc(BaseModel):
    category: list[str] = Field(default_factory=list)
    created_at: datetime
    importance: int = 0
    text: str

class PersonDoc(BaseModel):
    doc_id: str | None = Field(default=None, exclude=True)
    facts: list[PersonFactDoc] = Field(default_factory=list)
    user_id: str

person_mappings = {
    "dynamic": "strict",
    "properties": {
        "facts": {
            "type": "nested",
            "properties": {
                "category":   {"type": "keyword"},
                "created_at": {"type": "date"},
                "importance": {"type": "integer"},
                "text":       {"type": "text"},
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
        created_at = datetime.now(tz=timezone.utc)
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

    async def get_or_create_user_person_doc(self, user_id: str) -> PersonDoc:
        response = await self.es.search(index="persons", query={
            "term": {"user_id": user_id},
        })
        hits = response["hits"]["hits"]
        if hits:
            return PersonDoc(doc_id=hits[0]["_id"], **hits[0]["_source"])
        else:
            response = await self.es.index(index="persons", document={
                "user_id": user_id, "facts": []
            })
            return PersonDoc(doc_id=response["_id"], user_id=user_id)

    async def close(self):
        await self.es.close()