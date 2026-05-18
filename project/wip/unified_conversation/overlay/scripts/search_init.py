"""Overlay ES bootstrap.

Replaces upstream `scripts/search_init.py`. Creates the new `conversations` and
`turn-executions` indices alongside the unchanged `topics` index. The legacy
`context-partitions` index is *not* created — the overlay is a clean break.

Stop-word filter and `custom_query_analyzer` are preserved from upstream so
`topics` keeps its behavior; `conversations` and `turn-executions` reference
the standard analyzer in their mappings and don't depend on the custom
analyzer, but defining it index-wide is harmless.
"""
import asyncio
import logging

from dotenv import load_dotenv

from prokaryotes.search_v1 import get_elastic_search
from prokaryotes.search_v1.conversations import (
    CONVERSATIONS_INDEX,
    TURN_EXECUTIONS_INDEX,
    conversation_mappings,
    turn_execution_mappings,
)
from prokaryotes.search_v1.topics import topic_mappings
from prokaryotes.utils_v1.logging_utils import setup_logging

logger = logging.getLogger(__name__)

load_dotenv()
setup_logging()


async def sync_mappings(replicas: int = 0):
    schemas = {
        CONVERSATIONS_INDEX: conversation_mappings,
        TURN_EXECUTIONS_INDEX: turn_execution_mappings,
        "topics": topic_mappings,
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
                                    # TODO: Reference scikit-learn's english stop words
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
