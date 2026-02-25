import asyncio
import logging
from dotenv import load_dotenv

from prokaryotes.utils import setup_logging
from prokaryotes.search_v1 import (
    get_elastic_search,
    person_mappings,
)

logger = logging.getLogger(__name__)

load_dotenv()
setup_logging()

async def sync_mappings(replicas: int = 0):
    schemas = {
        "persons": person_mappings,
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
                "settings": {"number_of_replicas": replicas},
            })
            logger.info(f"Created index: {index_name}")
    await es.close()

asyncio.run(sync_mappings())
