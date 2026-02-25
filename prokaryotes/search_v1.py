import os
from datetime import datetime
from elasticsearch import AsyncElasticsearch
from pydantic import BaseModel, Field

class PersonFactDoc(BaseModel):
    category: list[str] = Field(default_factory=list)
    created_at: datetime
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
