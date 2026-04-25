import os

from elasticsearch import AsyncElasticsearch

from prokaryotes.search_v1.context_partitions import ContextPartitionSearcher
from prokaryotes.search_v1.named_entities import NamedEntitySearcher
from prokaryotes.search_v1.topics import TopicSearcher


class SearchClient(
    ContextPartitionSearcher,
    NamedEntitySearcher,
    TopicSearcher,
):
    def __init__(self):
        self._es: AsyncElasticsearch | None = None

    @property
    def es(self) -> AsyncElasticsearch | None:
        return self._es

    async def close(self):
        await self._es.close()

    def init_client(self):
        self._es = get_elastic_search()


def get_elastic_search() -> AsyncElasticsearch:
    uri = os.environ.get("ELASTIC_URI")
    if uri:
        return AsyncElasticsearch(uri)
    raise RuntimeError("Unable to initialize Elasticsearch client")
