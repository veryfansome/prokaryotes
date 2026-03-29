import os
from elasticsearch import AsyncElasticsearch

from prokaryotes.models_v1 import PersonContext
from prokaryotes.search_v1.facts import FactSearcher
from prokaryotes.search_v1.prompts import PromptSearcher
from prokaryotes.search_v1.responses import ResponseSearcher
from prokaryotes.search_v1.tool_calls import ToolCallSearcher
from prokaryotes.search_v1.topics import TopicSearcher

class SearchClient(
    FactSearcher,
    PromptSearcher,
    ResponseSearcher,
    ToolCallSearcher,
    TopicSearcher,
):
    def __init__(self):
        self._es: AsyncElasticsearch | None = None

    @property
    def es(self) -> AsyncElasticsearch | None:
        return self._es

    async def close(self):
        await self._es.close()

    async def get_user_context(
            self,
            full_name: str,
            user_id: int,
            match: str = None,
            match_emb: list[float] = None,
            min_facts_score: float = 0.5,
    ) -> PersonContext:
        facts = await self.search_facts(
            about=f"user:{user_id}",
            match=match,
            match_emb=match_emb,
            min_score=min_facts_score,
        )
        return PersonContext(facts=facts, name=full_name, user_id=user_id)

    def init_client(self):
        self._es = get_elastic_search()


def get_elastic_search() -> AsyncElasticsearch:
    uri = os.environ.get("ELASTIC_URI")
    if uri:
        return AsyncElasticsearch(uri)
    raise RuntimeError("Unable to initialize Elasticsearch client")
