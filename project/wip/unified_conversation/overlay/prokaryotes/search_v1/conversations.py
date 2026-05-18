"""Elasticsearch persistence for the unified conversation model.

Replaces `context_partitions.py`. Two indices:
- `conversations` — one doc per `Conversation` snapshot (compaction children and
  branch siblings alike). Retains `messages_json` even after `is_compacted=True`
  so the DAG-scoped assistant-message guardrail can read compacted ancestors'
  per-message identity at validation time.
- `turn-executions` — one doc per bot reply that involved tool calls. Keyed by
  `bot_message_source_id`. Looked up only when projecting a `Conversation` for
  the LLM, and only for the most recent turn(s) inside the raw window.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime

from elasticsearch import AsyncElasticsearch

from prokaryotes.conversation_v1.models import (
    Conversation,
    ConversationMessage,
    TurnExecution,
    TurnItem,
    compute_boundary_hash,
    compute_tail_hash,
    conversation_message_items,
)

logger = logging.getLogger(__name__)

CONVERSATIONS_INDEX = "conversations"
TURN_EXECUTIONS_INDEX = "turn-executions"
COMPACTION_STATE_COMMITTED = "committed"
COMPACTION_STATE_PENDING = "pending"

conversation_mappings = {
    "dynamic": "strict",
    "properties": {
        "snapshot_uuid": {"type": "keyword"},
        "conversation_uuid": {"type": "keyword"},
        "parent_snapshot_uuid": {"type": "keyword"},
        "bot_author_id": {"type": "keyword"},
        "compaction_state": {"type": "keyword"},
        "compaction_attempt_uuid": {"type": "keyword"},
        "is_compacted": {"type": "boolean"},
        "summary": {"type": "text", "analyzer": "standard"},
        "ancestor_summaries": {"type": "keyword", "index": False, "doc_values": False},
        "lifted_turn_items_json": {"type": "keyword", "index": False, "doc_values": False},
        "lifted_anchor_source_id": {"type": "keyword"},
        "messages_json": {"type": "keyword", "index": False, "doc_values": False},
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

turn_execution_mappings = {
    "dynamic": "strict",
    "properties": {
        "bot_message_source_id": {"type": "keyword"},
        "conversation_uuid": {"type": "keyword"},
        "items_json": {"type": "keyword", "index": False, "doc_values": False},
        "completed": {"type": "boolean"},
        "dt_created": {"type": "date"},
        "dt_modified": {"type": "date"},
    },
}


def _committed_or_legacy_clause() -> dict[str, object]:
    return {
        "bool": {
            "should": [
                {"term": {"compaction_state": COMPACTION_STATE_COMMITTED}},
                {"bool": {"must_not": [{"exists": {"field": "compaction_state"}}]}},
            ],
            "minimum_should_match": 1,
        }
    }


def _default_boundary_fields(conversation: Conversation) -> dict[str, object]:
    """Per-snapshot boundary fields (no parent walk).

    Used at non-compaction writes. The compactor overrides these with full-
    prefix-included values at the swap step — see `ConversationCompactor`.
    """
    non_deleted = conversation_message_items(conversation.messages)
    return {
        "raw_message_start_index": conversation.raw_message_start_index,
        "boundary_message_count": conversation.raw_message_start_index + len(non_deleted),
        "boundary_user_count": sum(1 for msg in non_deleted if msg.author_id != conversation.bot_author_id),
        "boundary_hash": compute_boundary_hash(non_deleted),
        "tail_hash": compute_tail_hash(non_deleted, conversation.bot_author_id),
    }


def _extract_message_content(messages: list[ConversationMessage]) -> str:
    return " ".join(msg.content for msg in conversation_message_items(messages) if msg.content)


def conversation_from_doc(conversation_uuid: str, doc: dict) -> Conversation | None:
    if doc.get("conversation_uuid") and doc["conversation_uuid"] != conversation_uuid:
        logger.warning(
            "Ignoring snapshot %s; belongs to conversation %s, not %s",
            doc.get("snapshot_uuid"),
            doc.get("conversation_uuid"),
            conversation_uuid,
        )
        return None
    if not doc.get("messages_json"):
        return None
    return Conversation(
        conversation_uuid=conversation_uuid,
        snapshot_uuid=doc["snapshot_uuid"],
        parent_snapshot_uuid=doc.get("parent_snapshot_uuid"),
        bot_author_id=doc["bot_author_id"],
        ancestor_summaries=doc.get("ancestor_summaries") or [],
        lifted_turn_items=lifted_turn_items_from_doc(doc),
        lifted_anchor_source_id=doc.get("lifted_anchor_source_id"),
        raw_message_start_index=doc.get("raw_message_start_index") or 0,
        messages=messages_from_doc(doc),
    )


def lifted_turn_items_from_doc(doc: dict) -> list[TurnItem]:
    lifted_json = doc.get("lifted_turn_items_json")
    if not lifted_json:
        return []
    payload = json.loads(lifted_json)
    return [TurnItem.model_validate(item) for item in payload.get("items", [])]


def messages_from_doc(doc: dict) -> list[ConversationMessage]:
    messages_json = doc.get("messages_json")
    if not messages_json:
        return []
    payload = json.loads(messages_json)
    return [ConversationMessage.model_validate(item) for item in payload.get("messages", [])]


def turn_execution_from_doc(doc: dict) -> TurnExecution | None:
    items_json = doc.get("items_json")
    if not items_json:
        return None
    payload = json.loads(items_json)
    return TurnExecution(
        conversation_uuid=doc["conversation_uuid"],
        bot_message_source_id=doc["bot_message_source_id"],
        items=[TurnItem.model_validate(item) for item in payload.get("items", [])],
        completed=doc.get("completed", False),
    )


class ConversationSearcher(ABC):
    @property
    @abstractmethod
    def es(self) -> AsyncElasticsearch:
        pass

    async def delete_turn_execution(self, bot_message_source_id: str) -> None:
        """Sweep — used when a `TurnExecution`'s owning bot run is fully tombstoned
        or the snapshot is compacted away."""
        try:
            await self.es.delete(index=TURN_EXECUTIONS_INDEX, id=bot_message_source_id)
        except Exception:
            logger.warning("Failed to delete turn-execution %s", bot_message_source_id, exc_info=True)

    async def rekey_turn_execution(self, old_id: str, new_id: str) -> None:
        """Move a `TurnExecution` doc from `old_id` to `new_id`.

        ES has no native id rename — implement as `index(new_id) + delete(old_id)`.
        Called when a bot `ConversationMessage` that owns a `TurnExecution` is
        tombstoned and the next non-tombstoned bot in the same consecutive run
        becomes the new owner.

        Silent no-op if the old doc doesn't exist (the run-walker may pass us a
        speculative re-key for an entry that was never persisted yet).
        """
        try:
            result = await self.es.get(index=TURN_EXECUTIONS_INDEX, id=old_id)
        except Exception:
            return
        doc = result["_source"]
        doc["bot_message_source_id"] = new_id
        doc["dt_modified"] = datetime.now(UTC).isoformat()
        try:
            await self.es.index(index=TURN_EXECUTIONS_INDEX, id=new_id, document=doc)
            await self.es.delete(index=TURN_EXECUTIONS_INDEX, id=old_id)
        except Exception:
            logger.warning(
                "Failed to rekey turn-execution %s -> %s",
                old_id,
                new_id,
                exc_info=True,
            )

    async def find_all_conversation_docs(self, conversation_uuid: str) -> list[dict]:
        """Return every conversation doc for `conversation_uuid` (compacted
        ancestors and branch siblings alike). Used by the DAG-scoped
        assistant-message guardrail to populate the known-source_id index.

        Cap is generous (10k) — a single dialogue's DAG sits in the tens of
        snapshots at most under realistic usage."""
        response = await self.es.search(
            index=CONVERSATIONS_INDEX,
            query={
                "bool": {
                    "must": [
                        {"term": {"conversation_uuid": conversation_uuid}},
                        _committed_or_legacy_clause(),
                    ]
                }
            },
            size=10_000,
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]

    async def find_conversation_by_tail_hash(self, conversation_uuid: str, tail_hash: str) -> dict | None:
        """Heuristic fallback for clients that didn't persist their `snapshot_uuid`:
        given a tail_hash computed from the last N non-bot messages, locate the
        nearest compacted ancestor without a full chain walk."""
        response = await self.es.search(
            index=CONVERSATIONS_INDEX,
            query={
                "bool": {
                    "must": [
                        {"term": {"conversation_uuid": conversation_uuid}},
                        {"term": {"tail_hash": tail_hash}},
                        {"term": {"is_compacted": True}},
                        _committed_or_legacy_clause(),
                    ]
                }
            },
            size=1,
        )
        hits = response["hits"]["hits"]
        return hits[0]["_source"] if hits else None

    async def find_latest_active_snapshot_uuid(self, conversation_uuid: str) -> str | None:
        """Return the `snapshot_uuid` of the most recently modified non-compacted
        snapshot for this `conversation_uuid`. Used when Redis has evicted the
        cache and the harness needs a head to start the chain walk from."""
        response = await self.es.search(
            index=CONVERSATIONS_INDEX,
            query={
                "bool": {
                    "must": [
                        {"term": {"conversation_uuid": conversation_uuid}},
                        {"term": {"is_compacted": False}},
                        _committed_or_legacy_clause(),
                    ]
                }
            },
            sort=[{"dt_modified": "desc"}],
            size=1,
        )
        hits = response["hits"]["hits"]
        return hits[0]["_source"]["snapshot_uuid"] if hits else None

    async def get_conversation(self, snapshot_uuid: str) -> dict | None:
        try:
            result = await self.es.get(index=CONVERSATIONS_INDEX, id=snapshot_uuid)
            return result["_source"]
        except Exception:
            return None

    async def get_turn_execution(self, bot_message_source_id: str) -> TurnExecution | None:
        try:
            result = await self.es.get(index=TURN_EXECUTIONS_INDEX, id=bot_message_source_id)
        except Exception:
            return None
        return turn_execution_from_doc(result["_source"])

    async def get_turn_executions(
        self, conversation_uuid: str, bot_message_source_ids: list[str]
    ) -> dict[str, TurnExecution]:
        """Batch lookup. Returns a dict keyed by `bot_message_source_id`."""
        if not bot_message_source_ids:
            return {}
        response = await self.es.search(
            index=TURN_EXECUTIONS_INDEX,
            query={
                "bool": {
                    "must": [
                        {"term": {"conversation_uuid": conversation_uuid}},
                        {"terms": {"bot_message_source_id": bot_message_source_ids}},
                    ]
                }
            },
            size=len(bot_message_source_ids),
        )
        out: dict[str, TurnExecution] = {}
        for hit in response["hits"]["hits"]:
            turn = turn_execution_from_doc(hit["_source"])
            if turn is not None:
                out[turn.bot_message_source_id] = turn
        return out

    async def put_conversation(
        self,
        conversation: Conversation,
        *,
        compaction_attempt_uuid: str | None = None,
        compaction_state: str = COMPACTION_STATE_COMMITTED,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        boundary_fields = _default_boundary_fields(conversation)
        doc = {
            "snapshot_uuid": conversation.snapshot_uuid,
            "conversation_uuid": conversation.conversation_uuid,
            "parent_snapshot_uuid": conversation.parent_snapshot_uuid,
            "bot_author_id": conversation.bot_author_id,
            "compaction_state": compaction_state,
            "compaction_attempt_uuid": compaction_attempt_uuid,
            "is_compacted": False,
            "summary": None,
            "ancestor_summaries": conversation.ancestor_summaries,
            "lifted_turn_items_json": json.dumps(
                {"items": [item.model_dump() for item in conversation.lifted_turn_items]}
            ),
            "lifted_anchor_source_id": conversation.lifted_anchor_source_id,
            "messages_json": json.dumps({"messages": [m.model_dump() for m in conversation.messages]}),
            "message_content": _extract_message_content(conversation.messages),
            "dt_created": now,
            "dt_modified": now,
            **boundary_fields,
        }
        await self.es.index(
            index=CONVERSATIONS_INDEX,
            id=conversation.snapshot_uuid,
            document=doc,
        )
        logger.info(
            "Stored conversation snapshot %s for conversation %s",
            conversation.snapshot_uuid,
            conversation.conversation_uuid,
        )

    async def put_turn_execution(self, turn: TurnExecution) -> None:
        now = datetime.now(UTC).isoformat()
        doc = {
            "bot_message_source_id": turn.bot_message_source_id,
            "conversation_uuid": turn.conversation_uuid,
            "items_json": json.dumps({"items": [item.model_dump() for item in turn.items]}),
            "completed": turn.completed,
            "dt_created": now,
            "dt_modified": now,
        }
        await self.es.index(
            index=TURN_EXECUTIONS_INDEX,
            id=turn.bot_message_source_id,
            document=doc,
        )

    async def search_conversations(self, conversation_uuid: str, query: str) -> list[dict]:
        response = await self.es.search(
            index=CONVERSATIONS_INDEX,
            query={
                "bool": {
                    "must": [
                        {"term": {"conversation_uuid": conversation_uuid}},
                        _committed_or_legacy_clause(),
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

    async def update_conversation(self, snapshot_uuid: str, **fields) -> None:
        fields["dt_modified"] = datetime.now(UTC).isoformat()
        await self.es.update(index=CONVERSATIONS_INDEX, id=snapshot_uuid, doc=fields)

    async def update_turn_execution(self, bot_message_source_id: str, **fields) -> None:
        fields["dt_modified"] = datetime.now(UTC).isoformat()
        await self.es.update(index=TURN_EXECUTIONS_INDEX, id=bot_message_source_id, doc=fields)
