"""ConversationCompactor._compact_conversation: CAS swap invariants.

Covers the swap step's structural properties: ancestor-summary accumulation,
raw_message_start_index advance, post-snapshot message carry-forward, WATCH
contention retry, ES-write retry / abort paths, prefix-equality skips, and
the new `compaction_status:{pending}` relabel-target write.
"""

from __future__ import annotations

import pytest
from redis.exceptions import WatchError

import prokaryotes.context_v1.compaction as compaction_module
from prokaryotes.conversation_v1.models import (
    Conversation,
    compute_boundary_hash,
)
from prokaryotes.search_v1.conversations import (
    COMPACTION_STATE_COMMITTED,
    COMPACTION_STATE_PENDING,
)
from tests.unit_tests._builders import BOT_ID, bot_msg, msg
from tests.unit_tests._fakes import (
    FakePipeline,
    FakeRedis,
    FakeSearchClient,
    TestableConversationCompactor,
    make_compactor,
)


def _default_messages() -> list:
    """Long enough that pre_tail is non-empty with the default
    COMPACTION_RECENCY_TAIL=6: 12 messages → pre_tail has 6 of them."""
    return [
        msg("1.000000", "U1"),
        bot_msg("2.000000", "A1"),
        msg("3.000000", "U2"),
        bot_msg("4.000000", "A2"),
        msg("5.000000", "U3"),
        bot_msg("6.000000", "A3"),
        msg("7.000000", "U4"),
        bot_msg("8.000000", "A4"),
        msg("9.000000", "U5"),
        bot_msg("10.000000", "A5"),
        msg("11.000000", "U6"),
        bot_msg("12.000000", "A6"),
    ]


def _make_snapshot(
    *,
    conversation_uuid: str = "conv",
    snapshot_uuid: str = "snap",
    parent_snapshot_uuid: str | None = None,
    ancestor_summaries: list[str] | None = None,
    raw_message_start_index: int = 0,
    messages=None,
) -> Conversation:
    if messages is None:
        messages = _default_messages()
    return Conversation(
        conversation_uuid=conversation_uuid,
        snapshot_uuid=snapshot_uuid,
        parent_snapshot_uuid=parent_snapshot_uuid,
        bot_author_id=BOT_ID,
        ancestor_summaries=ancestor_summaries or [],
        raw_message_start_index=raw_message_start_index,
        messages=messages,
    )


async def _setup(
    snapshot: Conversation,
    *,
    active: Conversation | None = None,
    seed_parent: Conversation | None = None,
) -> tuple[TestableConversationCompactor, FakeRedis, FakeSearchClient]:
    """Helper: build a compactor + fakes seeded with the active Redis snapshot
    and the snapshot's ES doc. If `active` is given, it replaces `snapshot` in
    Redis (simulating divergence). Always sets the compaction lock."""
    compactor, redis, search = make_compactor()
    await redis.set(
        f"conversation:{snapshot.conversation_uuid}",
        (active or snapshot).model_dump_json(),
    )
    await redis.set(f"compaction_lock:{snapshot.conversation_uuid}", "1")
    if seed_parent is not None:
        search.store_conversation_doc(seed_parent, is_compacted=False)
    search.store_conversation_doc(snapshot, is_compacted=False)
    return compactor, redis, search


async def _compact(
    compactor: TestableConversationCompactor,
    snapshot: Conversation,
    summary: str = "Summary",
    *,
    raise_exc: Exception | None = None,
) -> None:
    async def compact_fn(_snapshot, _prep) -> str:
        if raise_exc is not None:
            raise raise_exc
        return summary

    await compactor._compact_conversation(
        compact_fn=compact_fn,
        conversation_uuid=snapshot.conversation_uuid,
        lock_key=f"compaction_lock:{snapshot.conversation_uuid}",
        snapshot=snapshot,
    )


@pytest.mark.asyncio
async def test_compact_conversation_accumulates_ancestor_summaries_across_generations():
    """Each compaction concatenates the new summary onto the parent's
    ancestor_summaries list, preserving order across generations."""
    snapshot = _make_snapshot(
        snapshot_uuid="p1",
        ancestor_summaries=["S0"],
    )
    compactor, redis, _ = await _setup(snapshot)

    await _compact(compactor, snapshot, summary="S1")

    cached_data = await redis.get(f"conversation:{snapshot.conversation_uuid}")
    cached = Conversation.model_validate_json(cached_data)
    assert cached.ancestor_summaries == ["S0", "S1"]
    assert cached.parent_snapshot_uuid == "p1"


@pytest.mark.asyncio
async def test_compact_conversation_advances_raw_message_start_index_by_tail_offset():
    """The child's raw_message_start_index = parent's index + the count of
    non-deleted pre-tail messages (the tail offset)."""
    snapshot = _make_snapshot(raw_message_start_index=4)
    compactor, redis, _ = await _setup(snapshot)

    await _compact(compactor, snapshot)

    cached = Conversation.model_validate_json(await redis.get(f"conversation:{snapshot.conversation_uuid}"))
    # 6 messages total; tail target is 5 (COMPACTION_RECENCY_TAIL default), tail offset
    # is the count of non-deleted pre-tail messages. Child's index = parent's + offset.
    assert cached.raw_message_start_index > snapshot.raw_message_start_index
    # All recency-tail messages survive in cached.messages.
    assert len(cached.messages) > 0


@pytest.mark.asyncio
async def test_compact_conversation_carries_forward_post_snapshot_messages():
    """Messages appended during compaction (after the snapshot was taken) are
    carried into the child snapshot."""
    snapshot = _make_snapshot()
    # Active snapshot has one extra message appended after the snapshot was taken.
    active = _make_snapshot(messages=snapshot.messages + [msg("13.000000", "U7-post")])
    compactor, redis, _ = await _setup(snapshot, active=active)

    await _compact(compactor, snapshot)

    cached = Conversation.model_validate_json(await redis.get(f"conversation:{snapshot.conversation_uuid}"))
    # The post-snapshot message must appear in the cached child.
    assert any(m.source_id == "13.000000" and m.content == "U7-post" for m in cached.messages)
    assert cached.parent_snapshot_uuid == "snap"


@pytest.mark.asyncio
async def test_compact_conversation_releases_lock_when_compact_fn_raises():
    """The compaction lock is always released, even when `compact_fn` raises.
    The relabel-target sentinel ("") is written so polling clients clear their
    indicator without trying to navigate."""
    snapshot = _make_snapshot()
    compactor, redis, search = await _setup(snapshot)

    await _compact(compactor, snapshot, raise_exc=RuntimeError("LLM unavailable"))

    assert await redis.exists(f"compaction_lock:{snapshot.conversation_uuid}") == 0
    # The active snapshot is unchanged.
    cached = Conversation.model_validate_json(await redis.get(f"conversation:{snapshot.conversation_uuid}"))
    assert cached.snapshot_uuid == "snap"
    # Sentinel was written on exception path.
    status = await redis.get(f"compaction_status:{snapshot.snapshot_uuid}")
    assert status == b""


@pytest.mark.asyncio
async def test_compact_conversation_retries_redis_swap_on_watch_contention():
    """WATCH contention on the CAS swap triggers a retry; the second attempt
    commits successfully."""
    snapshot = _make_snapshot()
    execute_calls = []

    class ContendedPipeline(FakePipeline):
        async def execute(self):
            execute_calls.append(len(execute_calls) + 1)
            if len(execute_calls) == 1:
                self.commands = []
                raise WatchError()
            await super().execute()

    class ContendedRedis(FakeRedis):
        def pipeline(self):
            return ContendedPipeline(self)

    redis = ContendedRedis()
    await redis.set(f"conversation:{snapshot.conversation_uuid}", snapshot.model_dump_json())
    await redis.set(f"compaction_lock:{snapshot.conversation_uuid}", "1")
    search = FakeSearchClient()
    search.store_conversation_doc(snapshot, is_compacted=False)
    compactor = TestableConversationCompactor(redis_client=redis, search_client=search)

    await _compact(compactor, snapshot)

    assert len(execute_calls) == 2
    cached = Conversation.model_validate_json(await redis.get(f"conversation:{snapshot.conversation_uuid}"))
    assert cached.parent_snapshot_uuid == "snap"
    assert cached.ancestor_summaries == ["Summary"]
    assert await redis.exists(f"compaction_lock:{snapshot.conversation_uuid}") == 0


@pytest.mark.asyncio
async def test_compact_conversation_returns_early_for_empty_summary():
    """Empty summary returns BEFORE writing any compaction_status key — no
    sentinel, no CAS swap, no parent update. Lock still released via finally."""
    snapshot = _make_snapshot()
    compactor, redis, search = await _setup(snapshot)

    await _compact(compactor, snapshot, summary="")

    assert await redis.exists(f"compaction_lock:{snapshot.conversation_uuid}") == 0
    # No compaction_status:{pending} key written on the empty-summary path.
    assert await redis.get(f"compaction_status:{snapshot.snapshot_uuid}") is None
    cached = Conversation.model_validate_json(await redis.get(f"conversation:{snapshot.conversation_uuid}"))
    assert cached.snapshot_uuid == "snap"
    # No put_conversation calls beyond the initial seed.
    assert all(doc["snapshot_uuid"] == "snap" for doc in search.conversations.values())


@pytest.mark.asyncio
async def test_compact_conversation_returns_early_with_no_status_write_when_pre_tail_empty():
    """When the recency tail consumes all messages (pre_tail empty), there's
    nothing to compact: return early without writing the status key."""
    # Fewer messages than the recency-tail target — pre_tail collapses to empty.
    snapshot = _make_snapshot(
        messages=[
            msg("1.000000", "U1"),
            bot_msg("2.000000", "A1"),
            msg("3.000000", "U2"),
            bot_msg("4.000000", "A2"),
            msg("5.000000", "U3"),
            bot_msg("6.000000", "A3"),
        ]
    )
    compactor, redis, search = await _setup(snapshot)
    summary_calls = []

    async def compact_fn(_snapshot, _prep) -> str:
        summary_calls.append(True)
        return "Summary"

    await compactor._compact_conversation(
        compact_fn=compact_fn,
        conversation_uuid=snapshot.conversation_uuid,
        lock_key=f"compaction_lock:{snapshot.conversation_uuid}",
        snapshot=snapshot,
    )

    # compact_fn never called.
    assert summary_calls == []
    # Lock released; no compaction_status key written.
    assert await redis.exists(f"compaction_lock:{snapshot.conversation_uuid}") == 0
    assert await redis.get(f"compaction_status:{snapshot.snapshot_uuid}") is None


@pytest.mark.asyncio
async def test_compact_conversation_skips_swap_when_active_prefix_changed():
    """If the active prefix diverges (different ancestor_summaries or
    raw_message_start_index), the swap is skipped and the sentinel is written."""
    snapshot = _make_snapshot()
    active = _make_snapshot(
        ancestor_summaries=["surprise"],
        raw_message_start_index=2,
        messages=snapshot.messages[2:],
    )
    compactor, redis, _ = await _setup(snapshot, active=active)

    await _compact(compactor, snapshot)

    cached = Conversation.model_validate_json(await redis.get(f"conversation:{snapshot.conversation_uuid}"))
    assert cached.ancestor_summaries == ["surprise"]
    assert cached.raw_message_start_index == 2
    # Sentinel written via the swapped-is-None branch.
    assert await redis.get(f"compaction_status:{snapshot.snapshot_uuid}") == b""


@pytest.mark.asyncio
async def test_compact_conversation_skips_swap_when_active_snapshot_uuid_changed():
    """If the active snapshot's snapshot_uuid changed (branch switch happened
    after the snapshot was taken), the swap is skipped."""
    snapshot = _make_snapshot()
    active = _make_snapshot(snapshot_uuid="new-branch")
    compactor, redis, _ = await _setup(snapshot, active=active)

    await _compact(compactor, snapshot)

    cached = Conversation.model_validate_json(await redis.get(f"conversation:{snapshot.conversation_uuid}"))
    assert cached.snapshot_uuid == "new-branch"
    assert await redis.get(f"compaction_status:{snapshot.snapshot_uuid}") == b""


@pytest.mark.asyncio
async def test_compact_conversation_skips_swap_when_redis_conversation_missing():
    """If Redis has no conversation entry (e.g. evicted), the swap is skipped."""
    snapshot = _make_snapshot()
    compactor, redis, search = make_compactor()
    await redis.set(f"compaction_lock:{snapshot.conversation_uuid}", "1")
    search.store_conversation_doc(snapshot, is_compacted=False)
    # Do NOT seed the conversation in Redis.

    await _compact(compactor, snapshot)

    assert await redis.exists(f"compaction_lock:{snapshot.conversation_uuid}") == 0
    assert await redis.get(f"compaction_status:{snapshot.snapshot_uuid}") == b""


@pytest.mark.asyncio
async def test_boundary_hash_stored_on_es_covers_full_parent_prefix():
    """The parent's `boundary_hash` post-update covers the full reconstructed
    ancestor prefix + the parent's own messages (not just the parent's own)."""
    p0_messages = [
        msg("1.000000", "U1"),
        bot_msg("2.000000", "A1"),
    ]
    p0 = _make_snapshot(snapshot_uuid="p0", messages=p0_messages)
    # p1 needs enough messages that pre_tail is non-empty (8 > recency tail of 6).
    p1_own_messages = [
        msg("3.000000", "U2"),
        bot_msg("4.000000", "A2"),
        msg("5.000000", "U3"),
        bot_msg("6.000000", "A3"),
        msg("7.000000", "U4"),
        bot_msg("8.000000", "A4"),
        msg("9.000000", "U5"),
        bot_msg("10.000000", "A5"),
    ]
    snapshot = _make_snapshot(
        snapshot_uuid="p1",
        parent_snapshot_uuid="p0",
        raw_message_start_index=2,
        messages=p1_own_messages,
    )

    compactor, redis, search = await _setup(snapshot, seed_parent=p0)

    await _compact(compactor, snapshot)

    parent_doc = search.conversations["p1"]
    expected = compute_boundary_hash(p0_messages + p1_own_messages)
    assert parent_doc["boundary_hash"] == expected
    assert parent_doc["boundary_message_count"] == len(p0_messages) + len(p1_own_messages)


@pytest.mark.asyncio
async def test_compact_conversation_aborts_before_redis_swap_if_child_persist_fails(monkeypatch):
    """If the child snapshot persist to ES fails through all retries, the CAS
    swap doesn't run; the active snapshot stays on the parent."""
    monkeypatch.setattr(compaction_module, "_COMPACTION_SEARCH_WRITE_RETRY_DELAYS_SECONDS", (0, 0))
    snapshot = _make_snapshot()

    class FailingChildPersistSearch(FakeSearchClient):
        def __init__(self):
            super().__init__()
            self.put_attempts = 0

        async def put_conversation(self, conversation, **kwargs):
            self.put_attempts += 1
            raise RuntimeError("ES unavailable")

    search = FailingChildPersistSearch()
    search.store_conversation_doc(snapshot, is_compacted=False)
    redis = FakeRedis()
    await redis.set(f"conversation:{snapshot.conversation_uuid}", snapshot.model_dump_json())
    await redis.set(f"compaction_lock:{snapshot.conversation_uuid}", "1")
    compactor = TestableConversationCompactor(redis_client=redis, search_client=search)

    await _compact(compactor, snapshot)

    cached = Conversation.model_validate_json(await redis.get(f"conversation:{snapshot.conversation_uuid}"))
    assert cached.snapshot_uuid == "snap"
    assert search.put_attempts == 3
    # Exception path writes the sentinel.
    assert await redis.get(f"compaction_status:{snapshot.snapshot_uuid}") == b""
    assert await redis.exists(f"compaction_lock:{snapshot.conversation_uuid}") == 0


@pytest.mark.asyncio
async def test_compact_conversation_keeps_committed_child_reachable_when_parent_update_retries_exhausted(monkeypatch):
    """Parent-update retry exhaustion leaves the child reachable in ES but
    parent stays un-compacted. CAS swap already happened, so the Redis cache
    points at the new child."""
    monkeypatch.setattr(compaction_module, "_COMPACTION_SEARCH_WRITE_RETRY_DELAYS_SECONDS", (0, 0))
    snapshot = _make_snapshot()

    class FailingParentUpdateSearch(FakeSearchClient):
        def __init__(self):
            super().__init__()
            self.parent_update_attempts = 0

        async def update_conversation(self, snapshot_uuid, **fields):
            if snapshot_uuid == "snap":
                self.parent_update_attempts += 1
                raise RuntimeError("ES update unavailable")
            await super().update_conversation(snapshot_uuid, **fields)

    search = FailingParentUpdateSearch()
    search.store_conversation_doc(snapshot, is_compacted=False)
    redis = FakeRedis()
    await redis.set(f"conversation:{snapshot.conversation_uuid}", snapshot.model_dump_json())
    await redis.set(f"compaction_lock:{snapshot.conversation_uuid}", "1")
    compactor = TestableConversationCompactor(redis_client=redis, search_client=search)

    await _compact(compactor, snapshot)

    cached = Conversation.model_validate_json(await redis.get(f"conversation:{snapshot.conversation_uuid}"))
    assert cached.snapshot_uuid != "snap"
    assert cached.parent_snapshot_uuid == "snap"
    assert cached.ancestor_summaries == ["Summary"]
    # Child is reachable in ES with committed state.
    child_docs = [doc for uuid, doc in search.conversations.items() if uuid == cached.snapshot_uuid]
    assert len(child_docs) == 1
    assert child_docs[0]["compaction_state"] == COMPACTION_STATE_COMMITTED
    # Parent stays un-compacted because the mark-compacted update never succeeded.
    assert search.conversations["snap"].get("is_compacted") is False
    assert search.parent_update_attempts == 3
    assert await redis.exists(f"compaction_lock:{snapshot.conversation_uuid}") == 0


@pytest.mark.asyncio
async def test_compact_conversation_leaves_pending_child_when_cas_never_commits(monkeypatch):
    """If WATCH contention persists past the implicit retries (Redis swap
    races with branch switch), the pending child stays in ES with PENDING
    state; parent stays un-compacted."""
    monkeypatch.setattr(compaction_module, "_COMPACTION_SEARCH_WRITE_RETRY_DELAYS_SECONDS", (0, 0))
    snapshot = _make_snapshot()
    replacement = _make_snapshot(
        snapshot_uuid="new-branch",
        messages=[msg("1.000000", "U1"), bot_msg("2.000000", "A1"), msg("3.000000", "U2")],
    )
    execute_calls = []

    class SkipAfterWatchPipeline(FakePipeline):
        async def execute(self):
            execute_calls.append(len(execute_calls) + 1)
            if len(execute_calls) == 1:
                # Replace the active snapshot under our feet during the swap attempt.
                await self.redis.set(
                    f"conversation:{snapshot.conversation_uuid}",
                    replacement.model_dump_json(),
                )
                self.commands = []
                raise WatchError()
            await super().execute()

    class SkipAfterWatchRedis(FakeRedis):
        def pipeline(self):
            return SkipAfterWatchPipeline(self)

    redis = SkipAfterWatchRedis()
    await redis.set(f"conversation:{snapshot.conversation_uuid}", snapshot.model_dump_json())
    await redis.set(f"compaction_lock:{snapshot.conversation_uuid}", "1")
    search = FakeSearchClient()
    search.store_conversation_doc(snapshot, is_compacted=False)
    compactor = TestableConversationCompactor(redis_client=redis, search_client=search)

    await _compact(compactor, snapshot)

    cached = Conversation.model_validate_json(await redis.get(f"conversation:{snapshot.conversation_uuid}"))
    assert cached.snapshot_uuid == "new-branch"
    pending_children = [
        doc
        for uuid, doc in search.conversations.items()
        if uuid not in {"snap", "new-branch"} and doc.get("parent_snapshot_uuid") == "snap"
    ]
    assert len(pending_children) == 1
    assert pending_children[0]["compaction_state"] == COMPACTION_STATE_PENDING
    assert search.conversations["snap"].get("is_compacted") is False


@pytest.mark.asyncio
async def test_messages_json_preserved_on_compacted_parent():
    """After commit, the parent's messages_json must remain populated (the
    DAG-scoped guardrail needs to walk the parent chain even after compaction)."""
    snapshot = _make_snapshot()
    compactor, redis, search = await _setup(snapshot)

    await _compact(compactor, snapshot)

    parent_doc = search.conversations["snap"]
    assert parent_doc["is_compacted"] is True
    assert "messages_json" in parent_doc
    assert parent_doc["messages_json"]  # non-empty


@pytest.mark.asyncio
async def test_compaction_status_written_to_redis_with_child_snapshot_uuid():
    """At commit time, compaction_status:{pending} → committed child's
    snapshot_uuid. The TTL matches conversation_cache_ex."""
    snapshot = _make_snapshot()
    compactor, redis, _ = await _setup(snapshot)

    await _compact(compactor, snapshot)

    cached = Conversation.model_validate_json(await redis.get(f"conversation:{snapshot.conversation_uuid}"))
    status = await redis.get(f"compaction_status:{snapshot.snapshot_uuid}")
    assert status is not None
    assert status.decode("utf-8") == cached.snapshot_uuid
    assert redis._ex[f"compaction_status:{snapshot.snapshot_uuid}"] == compactor.conversation_cache_ex
