import os
import pathlib

import prokaryotes

# Allow unchanged sibling modules (topics.py) in the real package to fall through.
# Without this, the overlay's __init__.py would shadow the real search_v1 entirely.
_HERE = pathlib.Path(__file__).resolve().parent
for _parent_path in prokaryotes.__path__:
    _candidate = pathlib.Path(_parent_path).resolve() / "search_v1"
    if _candidate != _HERE and _candidate.is_dir() and str(_candidate) not in __path__:
        __path__.append(str(_candidate))

from elasticsearch import AsyncElasticsearch  # noqa: E402

from prokaryotes.search_v1.conversations import ConversationSearcher  # noqa: E402
from prokaryotes.search_v1.topics import TopicSearcher  # noqa: E402


class SearchClient(
    ConversationSearcher,
    TopicSearcher,
):
    def __init__(self):
        self._es: AsyncElasticsearch | None = None

    async def close(self):
        await self._es.close()

    @property
    def es(self) -> AsyncElasticsearch | None:
        return self._es

    def init_client(self):
        self._es = get_elastic_search()


def get_elastic_search() -> AsyncElasticsearch:
    uri = os.environ.get("ELASTIC_URI")
    if uri:
        return AsyncElasticsearch(uri)
    raise RuntimeError("Unable to initialize Elasticsearch client")
