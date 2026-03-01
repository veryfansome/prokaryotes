import asyncio
import logging
from dotenv import load_dotenv

from prokaryotes.utils import setup_logging
from prokaryotes.search_v1 import (
    fact_mappings,
    get_elastic_search,
    question_mappings,
)

logger = logging.getLogger(__name__)

load_dotenv()
setup_logging()

async def sync_mappings(replicas: int = 0):
    schemas = {
        "facts": fact_mappings,
        "questions": question_mappings,
    }
    es = get_elastic_search()
    for index_name, mappings in schemas.items():
        if await es.indices.exists(index=index_name):
            await es.indices.put_mapping(index=index_name, body=mappings)
            logger.info(f"Updated mapping for index: {index_name}")
            await es.indices.put_settings(index=index_name, body={
                "index": {
                    "number_of_replicas": replicas,
                },
            })
            logger.info(f"Updated settings for index: {index_name}")
        else:
            await es.indices.create(index=index_name, body={
                "mappings": mappings,
                "settings": {
                    "analysis": {
                        "filter": {
                            "extended_stop_filter": {
                                "type": "stop",
                                "stopwords": [
                                    # https://docs.opensearch.org/latest/analyzers/token-filters/stop
                                    "_english_",
                                    "he", "his",
                                    "her", "hers",
                                    "theirs",  # _english_ includes: they, their
                                    "its",  # _english_ includes: it
                                ]
                            }
                        },
                        "analyzer": {
                            "custom_query_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "extended_stop_filter"]
                            }
                        }
                    },
                    "number_of_replicas": replicas,
                },
            })
            logger.info(f"Created index: {index_name}")
    await es.close()

asyncio.run(sync_mappings())
