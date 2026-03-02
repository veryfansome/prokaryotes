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
                                    "ain't",
                                    "all",
                                    "am",
                                    "any", "anybody", "anyone", "anything",
                                    "aren't",
                                    "been", "being",
                                    "both",
                                    "can", "can't",
                                    "could", "couldn't",
                                    "did", "didn't",
                                    "do", "don't",
                                    "does", "doesn't",
                                    "done",
                                    "each",
                                    "every", "everybody", "everyone", "everything",
                                    "few",
                                    "had", "hadn't",
                                    "has", "hasn't",
                                    "have", "haven't",
                                    "he", "he'd", "he'll", "he's",
                                    "her", "hers",
                                    "him",
                                    "his",
                                    "how",
                                    "i", "i'd", "i'll", "i'm", "i've",
                                    "isn't",
                                    "it'll", "it's", "its", # _english_ includes: it
                                    "may",
                                    "me",
                                    "might", "mightn't",
                                    "mine",
                                    "more",
                                    "most",
                                    "my",
                                    "must", "mustn't",
                                    "nobody", "noone", "nothing",
                                    "other",
                                    "ought",
                                    "our", "ours",
                                    "shall",
                                    "she", "she'd", "she'll", "she's",
                                    "should", "shouldn't",
                                    "some", "somebody", "someone", "something",
                                    "theirs",  # _english_ includes: they, their
                                    "them",
                                    "they'd", "they'll", "they're", "they've",
                                    "we", "we'd", "we'll", "we've",
                                    "were", "weren't",
                                    "what", "what's",
                                    "when", "when's",
                                    "where", "where's",
                                    "which",
                                    "who", "who'd", "who'll", "who's",
                                    "why",
                                    "won't",
                                    "would", "wouldn't",
                                    "you", "you'd", "you'll", "you've",
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
