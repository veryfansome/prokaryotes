"""Hermetic fakes for Redis and the search client.

Used by syncer/compactor unit tests so we can exercise multi-tier load,
chain rebuild, prefix split, branch divergence, and tombstone re-keying
without standing up Docker. The fakes track every call site the production
syncer/compactor reaches (`get_conversation`, `put_conversation`,
`update_conversation`, `get_turn_executions`, `put_turn_execution`,
`rekey_turn_execution`, `delete_turn_execution`) and the Redis verbs the
syncer uses (`get`, `set`, `delete`, `exists`).

The compactor's CAS pipeline (`watch` / `multi` / `execute`) is intentionally
out of scope here — those paths are exercised by the Tier-B integration suite
against real Redis.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from prokaryotes.context_v1.compaction import ConversationCompactor
from prokaryotes.context_v1.conversation_sync import ConversationSyncer
from prokaryotes.conversation_v1.models import Conversation, TurnExecution


class FakeRedis:
    """In-memory dict-shaped fake for the subset of `redis.asyncio.Redis` the syncer uses.

    Values are stored as the bytes the real client would return on `get`; `set` accepts
    either str or bytes. `ex` is recorded but not enforced (tests don't measure TTL).

    `pipeline()` returns a `FakePipeline` supporting watch/multi/execute for the
    compactor's CAS swap. The pipeline only models the verbs the compactor uses;
    fanned-out usage from production code paths the compactor doesn't touch is
    intentionally out of scope.
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}
        self._ex: dict[str, int | None] = {}
        # Recording hooks.
        self.set_calls: list[tuple[str, bytes, int | None, bool]] = []
        self.delete_calls: list[tuple[str, ...]] = []

    async def delete(self, *keys: str) -> int:
        self.delete_calls.append(tuple(keys))
        removed = 0
        for key in keys:
            if key in self._store:
                self._store.pop(key)
                self._ex.pop(key, None)
                removed += 1
        return removed

    async def exists(self, key: str) -> int:
        return 1 if key in self._store else 0

    async def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)

    async def set(
        self,
        key: str,
        value: str | bytes,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        if nx and key in self._store:
            return None
        if isinstance(value, str):
            value = value.encode("utf-8")
        self._store[key] = value
        self._ex[key] = ex
        self.set_calls.append((key, value, ex, nx))
        return True


class FakePipeline:
    """Minimal `redis.asyncio.client.Pipeline` for the compactor's CAS swap.

    Tests inject WATCH contention by subclassing and overriding `execute`. The
    pipeline doesn't enforce a true MULTI/EXEC semantic — `set` queues into
    `commands` after `multi()`, and `execute()` flushes them to the underlying
    FakeRedis. Use `_force_watch_error_on_execute=N` to fail the next N
    `execute()` calls with WatchError before applying queued writes.
    """

    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.commands: list[tuple[str, bytes | str, int | None]] = []
        self.watched_key: str | None = None
        self.execute_calls = 0

    async def __aenter__(self) -> FakePipeline:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def execute(self) -> None:
        self.execute_calls += 1
        for key, value, ex in self.commands:
            await self.redis.set(key, value, ex=ex)
        self.commands = []

    async def get(self, key: str) -> bytes | None:
        return await self.redis.get(key)

    def multi(self) -> None:
        self.commands = []

    async def reset(self) -> None:
        self.commands = []

    def set(self, key: str, value: str | bytes, ex: int | None = None) -> None:
        self.commands.append((key, value, ex))

    async def watch(self, key: str) -> None:
        self.watched_key = key


class FakeSearchClient:
    """In-memory fake for the overlay's `SearchClient` surface.

    Stores `conversations` keyed by `snapshot_uuid` and `turn_executions` keyed by
    `bot_message_source_id`. Document shapes mirror what the real searcher persists
    (so `conversation_from_doc` / `turn_execution_from_doc` can round-trip them).
    """

    def __init__(self) -> None:
        self.conversations: dict[str, dict[str, Any]] = {}
        self.turn_executions: dict[str, dict[str, Any]] = {}
        # Test hooks — count calls so re-keying tests can assert on ES operations.
        self.delete_turn_execution_calls: list[str] = []
        self.put_turn_execution_calls: list[TurnExecution] = []
        self.rekey_turn_execution_calls: list[tuple[str, str]] = []

    async def delete_turn_execution(self, bot_message_source_id: str) -> None:
        self.delete_turn_execution_calls.append(bot_message_source_id)
        self.turn_executions.pop(bot_message_source_id, None)

    async def find_all_conversation_docs(self, conversation_uuid: str) -> list[dict[str, Any]]:
        return [doc for doc in self.conversations.values() if doc.get("conversation_uuid") == conversation_uuid]

    async def get_conversation(self, snapshot_uuid: str) -> dict[str, Any] | None:
        return self.conversations.get(snapshot_uuid)

    async def get_turn_execution(self, bot_message_source_id: str) -> TurnExecution | None:
        from prokaryotes.search_v1.conversations import turn_execution_from_doc

        doc = self.turn_executions.get(bot_message_source_id)
        if doc is None:
            return None
        return turn_execution_from_doc(doc)

    async def get_turn_executions(
        self,
        conversation_uuid: str,
        bot_message_source_ids: list[str],
    ) -> dict[str, TurnExecution]:
        from prokaryotes.search_v1.conversations import turn_execution_from_doc

        out: dict[str, TurnExecution] = {}
        for sid in bot_message_source_ids:
            doc = self.turn_executions.get(sid)
            if doc is None:
                continue
            if doc.get("conversation_uuid") != conversation_uuid:
                continue
            turn = turn_execution_from_doc(doc)
            if turn is not None:
                out[sid] = turn
        return out

    async def put_conversation(
        self,
        conversation: Conversation,
        *,
        compaction_attempt_uuid: str | None = None,
        compaction_state: str = "committed",
    ) -> None:
        self.conversations[conversation.snapshot_uuid] = self._build_conversation_doc(
            conversation,
            compaction_attempt_uuid=compaction_attempt_uuid,
            compaction_state=compaction_state,
        )

    async def put_turn_execution(self, turn: TurnExecution) -> None:
        self.put_turn_execution_calls.append(turn)
        now = datetime.now(UTC).isoformat()
        self.turn_executions[turn.bot_message_source_id] = {
            "bot_message_source_id": turn.bot_message_source_id,
            "conversation_uuid": turn.conversation_uuid,
            "items_json": json.dumps({"items": [item.model_dump() for item in turn.items]}),
            "completed": turn.completed,
            "dt_created": now,
            "dt_modified": now,
        }

    async def rekey_turn_execution(self, old_id: str, new_id: str) -> None:
        """Move a TurnExecution doc from `old_id` to `new_id`.

        Mirrors the production contract: ES has no native id rename, so the
        implementation must put the doc under the new id and delete the old id.
        """
        self.rekey_turn_execution_calls.append((old_id, new_id))
        if old_id not in self.turn_executions:
            return
        doc = dict(self.turn_executions[old_id])
        doc["bot_message_source_id"] = new_id
        self.turn_executions[new_id] = doc
        self.turn_executions.pop(old_id, None)

    async def update_conversation(self, snapshot_uuid: str, **fields: Any) -> None:
        doc = self.conversations.get(snapshot_uuid)
        if doc is None:
            return
        doc.update(fields)
        doc["dt_modified"] = datetime.now(UTC).isoformat()

    def store_conversation_doc(
        self,
        conversation: Conversation,
        *,
        is_compacted: bool = False,
        summary: str | None = None,
        boundary_hash: str | None = None,
        boundary_message_count: int | None = None,
        boundary_user_count: int | None = None,
        tail_hash: str | None = None,
    ) -> None:
        """Test-only setup helper. Persists a Conversation as if produced by the
        compactor — `is_compacted=True` for parent snapshots, with `summary` set.

        Optional boundary fields mirror what the real compactor writes onto the
        parent snapshot at commit time (`_compact_conversation`). They're
        required for chain-rebuild validation in `_rebuild_from_chain`.
        """
        doc = self._build_conversation_doc(conversation, compaction_state="committed")
        doc["is_compacted"] = is_compacted
        doc["summary"] = summary
        if boundary_hash is not None:
            doc["boundary_hash"] = boundary_hash
        if boundary_message_count is not None:
            doc["boundary_message_count"] = boundary_message_count
        if boundary_user_count is not None:
            doc["boundary_user_count"] = boundary_user_count
        if tail_hash is not None:
            doc["tail_hash"] = tail_hash
        self.conversations[conversation.snapshot_uuid] = doc

    @staticmethod
    def _build_conversation_doc(
        conversation: Conversation,
        *,
        compaction_attempt_uuid: str | None = None,
        compaction_state: str = "committed",
    ) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        return {
            "snapshot_uuid": conversation.snapshot_uuid,
            "conversation_uuid": conversation.conversation_uuid,
            "parent_snapshot_uuid": conversation.parent_snapshot_uuid,
            "bot_author_id": conversation.bot_author_id,
            "compaction_state": compaction_state,
            "compaction_attempt_uuid": compaction_attempt_uuid,
            "is_compacted": False,
            "summary": None,
            "ancestor_summaries": list(conversation.ancestor_summaries),
            "lifted_turn_items_json": json.dumps(
                {"items": [item.model_dump() for item in conversation.lifted_turn_items]}
            ),
            "lifted_anchor_source_id": conversation.lifted_anchor_source_id,
            "messages_json": json.dumps({"messages": [m.model_dump() for m in conversation.messages]}),
            "raw_message_start_index": conversation.raw_message_start_index,
            "dt_created": now,
            "dt_modified": now,
        }


class TestableConversationSyncer(ConversationSyncer):
    """Concrete `ConversationSyncer` wired to fakes for unit tests."""

    __test__ = False  # not a pytest test class

    def __init__(
        self,
        *,
        redis_client: FakeRedis,
        search_client: FakeSearchClient,
        conversation_cache_ex: int = 60 * 60 * 24 * 7,
    ) -> None:
        self._redis_client = redis_client
        self._search_client = search_client
        self._conversation_cache_ex = conversation_cache_ex

    @property
    def conversation_cache_ex(self) -> int:
        return self._conversation_cache_ex

    @property
    def redis_client(self) -> FakeRedis:  # type: ignore[override]
        return self._redis_client

    @property
    def search_client(self) -> FakeSearchClient:  # type: ignore[override]
        return self._search_client


def make_syncer() -> tuple[TestableConversationSyncer, FakeRedis, FakeSearchClient]:
    redis = FakeRedis()
    search = FakeSearchClient()
    syncer = TestableConversationSyncer(redis_client=redis, search_client=search)
    return syncer, redis, search


class TestableConversationCompactor(ConversationCompactor):
    """Concrete `ConversationCompactor` wired to fakes for unit tests."""

    __test__ = False  # not a pytest test class

    def __init__(
        self,
        *,
        redis_client: FakeRedis,
        search_client: FakeSearchClient,
        conversation_cache_ex: int = 60 * 60 * 24 * 7,
    ) -> None:
        self._redis_client = redis_client
        self._search_client = search_client
        self._conversation_cache_ex = conversation_cache_ex

    @property
    def conversation_cache_ex(self) -> int:
        return self._conversation_cache_ex

    @property
    def redis_client(self) -> FakeRedis:  # type: ignore[override]
        return self._redis_client

    @property
    def search_client(self) -> FakeSearchClient:  # type: ignore[override]
        return self._search_client


def make_compactor() -> tuple[TestableConversationCompactor, FakeRedis, FakeSearchClient]:
    redis = FakeRedis()
    search = FakeSearchClient()
    compactor = TestableConversationCompactor(redis_client=redis, search_client=search)
    return compactor, redis, search
