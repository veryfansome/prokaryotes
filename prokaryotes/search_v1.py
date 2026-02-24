import logging
import os
from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)

def get_async_elastic_search() -> AsyncElasticsearch:
    elastic_uri = os.environ.get("ELASTIC_URI")
    if elastic_uri:
        return AsyncElasticsearch(elastic_uri)
    raise RuntimeError("Unable to initialize Elasticsearch client")
