import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime

from elasticsearch import AsyncElasticsearch

from prokaryotes.api_v1.models import (
    ContextPartition,
    ContextPartitionItem,
    compute_boundary_hash,
    compute_tail_hash,
    conversation_message_items,
)

logger = logging.getLogger(__name__)

INDEX = "context-partitions"

context_partition_mappings = {
    "dynamic": "strict",
    "properties": {
        "partition_uuid": {"type": "keyword"},
        "conversation_uuid": {"type": "keyword"},
        "parent_partition_uuid": {"type": "keyword"},
        "is_compacted": {"type": "boolean"},
        "summary": {"type": "text", "analyzer": "standard"},
        "ancestor_summaries": {"type": "keyword", "index": False, "doc_values": False},
        "items_json": {"type": "keyword", "index": False, "doc_values": False},
        "message_content": {"type": "text", "analyzer": "standard"},
        "raw_message_start_index": {"type": "integer"},
        "boundary_message_count": {"type": "integer"},
        "boundary_user_count": {"type": "integer"},
        "boundary_hash": {"type": "keyword"},
        "tail_hash": {"type": "keyword"},
        "dt_created": {"type": "date"},
        "dt_modified": {"type": "date"},
    },
}


def _extract_message_content(items: list[ContextPartitionItem]) -> str:
    return " ".join(
        item.content
        for item in conversation_message_items(items)
        if item.content
    )


def _default_boundary_fields(partition: ContextPartition) -> dict[str, object]:
    message_items = conversation_message_items(partition.items)
    boundary_message_count = partition.raw_message_start_index + len(message_items)
    return {
        "raw_message_start_index": partition.raw_message_start_index,
        "boundary_message_count": boundary_message_count,
        "boundary_user_count": sum(1 for item in message_items if item.role == "user"),
        "boundary_hash": compute_boundary_hash(message_items),
        "tail_hash": compute_tail_hash(message_items),
    }


def items_from_doc(doc: dict) -> list[ContextPartitionItem]:
    items_json = doc.get("items_json")
    if not items_json:
        return []
    return [ContextPartitionItem.model_validate(item) for item in json.loads(items_json)["items"]]


def partition_from_doc(doc: dict, conversation_uuid: str) -> ContextPartition | None:
    if doc.get("conversation_uuid") and doc["conversation_uuid"] != conversation_uuid:
        logger.warning(
            "Ignoring partition %s because it belongs to conversation %s, not %s",
            doc.get("partition_uuid"),
            doc.get("conversation_uuid"),
            conversation_uuid,
        )
        return None
    if not doc.get("items_json"):
        return None
    return ContextPartition(
        conversation_uuid=conversation_uuid,
        partition_uuid=doc["partition_uuid"],
        parent_partition_uuid=doc.get("parent_partition_uuid"),
        ancestor_summaries=doc.get("ancestor_summaries") or [],
        raw_message_start_index=doc.get("raw_message_start_index") or 0,
        items=items_from_doc(doc),
    )


class ContextPartitionSearcher(ABC):
    @property
    @abstractmethod
    def es(self) -> AsyncElasticsearch:
        pass

    async def get_partition(self, partition_uuid: str) -> dict | None:
        try:
            result = await self.es.get(index=INDEX, id=partition_uuid)
            return result["_source"]
        except Exception:
            return None

    async def put_partition(self, partition: ContextPartition) -> None:
        now = datetime.now(UTC).isoformat()
        boundary_fields = _default_boundary_fields(partition)
        doc = {
            "partition_uuid": partition.partition_uuid,
            "conversation_uuid": partition.conversation_uuid,
            "parent_partition_uuid": partition.parent_partition_uuid,
            "is_compacted": False,
            "summary": None,
            "ancestor_summaries": partition.ancestor_summaries,
            "items_json": partition.model_dump_json(include={"items"}),
            "message_content": _extract_message_content(partition.items),
            "dt_created": now,
            "dt_modified": now,
            **boundary_fields,
        }
        await self.es.index(index=INDEX, id=partition.partition_uuid, document=doc)
        logger.info(
            "Stored partition %s for conversation %s",
            partition.partition_uuid,
            partition.conversation_uuid,
        )

    async def update_partition(self, partition_uuid: str, **fields) -> None:
        fields["dt_modified"] = datetime.now(UTC).isoformat()
        await self.es.update(index=INDEX, id=partition_uuid, doc=fields)

    # tail_hash is stored on every compacted partition but not consulted during normal reconciliation,
    # which uses boundary_hash for exact prefix validation instead. find_partition_by_tail_hash is
    # the hook for a future heuristic-fallback path: given a conversation whose partition_uuid is
    # unknown (e.g., a client that did not persist it), a tail_hash computed from the last N user
    # messages could locate the nearest compacted ancestor without a full chain walk.
    async def find_partition_by_tail_hash(self, conversation_uuid: str, tail_hash: str) -> dict | None:
        response = await self.es.search(
            index=INDEX,
            query={
                "bool": {
                    "must": [
                        {"term": {"conversation_uuid": conversation_uuid}},
                        {"term": {"tail_hash": tail_hash}},
                        {"term": {"is_compacted": True}},
                    ]
                }
            },
            size=1,
        )
        hits = response["hits"]["hits"]
        return hits[0]["_source"] if hits else None

    async def search_partitions(self, conversation_uuid: str, query: str) -> list[dict]:
        response = await self.es.search(
            index=INDEX,
            query={
                "bool": {
                    "must": [
                        {"term": {"conversation_uuid": conversation_uuid}},
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["summary", "message_content"],
                            }
                        },
                    ]
                }
            },
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]
