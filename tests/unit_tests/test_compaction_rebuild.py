"""ConversationSyncer._rebuild_from_chain + _walk_snapshot_chain.

Tier-3 load step: walks the snapshot DAG from the client's snapshot_uuid back to a compacted ancestor whose
boundary_hash matches a prefix of incoming, then builds a Conversation rooted there. Safety: chain walk stops on
conversation_uuid mismatch, cycles, or missing intermediate docs.

Also covers HarnessBase.stream_and_finalize's handshake + compaction_pending ordering and the
duplicate-compaction-when-lock-held skip.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from prokaryotes.context_v1.conversation_sync import (
    SourceIdAssignment,
    SyncResult,
    _PartialMessage,
)
from prokaryotes.conversation_v1.models import (
    ConversationMessage,
    compute_boundary_hash,
    compute_tail_hash,
)
from prokaryotes.harness_v1.base import HarnessBase
from tests.unit_tests._builders import BOT_ID, bot_msg, msg
from tests.unit_tests._fakes import (
    FakeRedis,
    FakeSearchClient,
    make_syncer,
)


def _make_partial(messages: list[ConversationMessage]) -> list[_PartialMessage]:
    """Translate ConversationMessages → _PartialMessages preserving source_ids."""
    return [
        _PartialMessage(
            author_id=m.author_id,
            content=m.content,
            client_index=i,
            source_id=m.source_id,
            display_name=m.display_name,
        )
        for i, m in enumerate(messages)
    ]


def _compacted_doc(
    *,
    boundary: list[ConversationMessage],
    snapshot_uuid: str,
    parent_snapshot_uuid: str | None,
    summary: str,
    conversation_uuid: str = "conv",
    bot_author_id: str = BOT_ID,
    boundary_message_count_override: int | None = None,
    boundary_hash_override: str | None = None,
) -> dict[str, Any]:
    """Build a FakeSearchClient-compatible compacted doc with valid boundary fields."""
    return {
        "snapshot_uuid": snapshot_uuid,
        "conversation_uuid": conversation_uuid,
        "parent_snapshot_uuid": parent_snapshot_uuid,
        "bot_author_id": bot_author_id,
        "compaction_state": "committed",
        "is_compacted": True,
        "summary": summary,
        "ancestor_summaries": [],
        "boundary_hash": boundary_hash_override or compute_boundary_hash(boundary),
        "boundary_message_count": boundary_message_count_override or len(boundary),
        "boundary_user_count": sum(1 for m in boundary if m.author_id != bot_author_id),
        "tail_hash": compute_tail_hash(boundary, bot_author_id),
        "messages_json": json.dumps({"messages": [m.model_dump() for m in boundary]}),
        "lifted_turn_items_json": json.dumps({"items": []}),
        "lifted_anchor_source_id": None,
        "raw_message_start_index": 0,
    }


def _seed_doc(search: FakeSearchClient, doc: dict[str, Any]) -> None:
    search.conversations[doc["snapshot_uuid"]] = doc


# -----------------------------------------------------------------------------
# _walk_snapshot_chain — safety tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_walk_snapshot_chain_stops_on_conversation_uuid_mismatch():
    syncer, _, search = make_syncer()
    _seed_doc(
        search, _compacted_doc(boundary=[msg("1", "U1")], snapshot_uuid="p1", parent_snapshot_uuid="p0", summary="S1")
    )
    _seed_doc(
        search,
        _compacted_doc(
            boundary=[msg("1", "U1")],
            snapshot_uuid="p0",
            parent_snapshot_uuid=None,
            summary="S0",
            conversation_uuid="OTHER",
        ),
    )

    chain = await syncer._walk_snapshot_chain(conversation_uuid="conv", snapshot_uuid="p1")

    # The walk stops when the parent doc belongs to a different conversation_uuid.
    assert [d["snapshot_uuid"] for d in chain] == ["p1"]


@pytest.mark.asyncio
async def test_walk_snapshot_chain_stops_on_cycle():
    syncer, _, search = make_syncer()
    # p1 → p0 → p1 cycle.
    _seed_doc(
        search, _compacted_doc(boundary=[msg("1", "U1")], snapshot_uuid="p1", parent_snapshot_uuid="p0", summary="S1")
    )
    _seed_doc(
        search, _compacted_doc(boundary=[msg("1", "U1")], snapshot_uuid="p0", parent_snapshot_uuid="p1", summary="S0")
    )

    chain = await syncer._walk_snapshot_chain(conversation_uuid="conv", snapshot_uuid="p1")

    # Walk follows the cycle once and stops on seen-set hit.
    assert [d["snapshot_uuid"] for d in chain] == ["p1", "p0"]


@pytest.mark.asyncio
async def test_walk_snapshot_chain_stops_when_intermediate_doc_missing():
    syncer, _, search = make_syncer()
    _seed_doc(
        search,
        _compacted_doc(boundary=[msg("1", "U1")], snapshot_uuid="p1", parent_snapshot_uuid="p0-missing", summary="S1"),
    )
    # p0-missing has no ES doc; the walk halts.

    chain = await syncer._walk_snapshot_chain(conversation_uuid="conv", snapshot_uuid="p1")

    assert [d["snapshot_uuid"] for d in chain] == ["p1"]


# -----------------------------------------------------------------------------
# _rebuild_from_chain — semantic tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rebuild_from_chain_assembles_two_generation_summaries_in_order():
    """Two compacted ancestors with valid boundaries → ancestor_summaries populated outer-first."""
    syncer, _, search = make_syncer()
    p0_boundary = [msg("1", "U1"), bot_msg("2", "A1")]
    p1_boundary = p0_boundary + [msg("3", "U2"), bot_msg("4", "A2")]
    _seed_doc(search, _compacted_doc(boundary=p0_boundary, snapshot_uuid="p0", parent_snapshot_uuid=None, summary="S0"))
    _seed_doc(search, _compacted_doc(boundary=p1_boundary, snapshot_uuid="p1", parent_snapshot_uuid="p0", summary="S1"))

    incoming = p1_boundary + [msg("5", "U3")]
    rebuilt = await syncer._rebuild_from_chain(
        conversation_uuid="conv",
        snapshot_uuid="p1",
        bot_author_id=BOT_ID,
        partial=_make_partial(incoming),
        head_doc=None,
    )

    assert rebuilt.parent_snapshot_uuid == "p1"
    assert rebuilt.ancestor_summaries == ["S0", "S1"]
    assert rebuilt.raw_message_start_index == 4
    assert [m.source_id for m in rebuilt.messages] == ["5"]


@pytest.mark.asyncio
async def test_rebuild_from_chain_does_not_inject_stale_summary_without_valid_boundary():
    """If an ancestor's boundary_hash doesn't match incoming, its summary is skipped. With no matching
    ancestor, fall back to a fresh Conversation (no ancestors)."""
    syncer, _, search = make_syncer()
    _seed_doc(
        search,
        _compacted_doc(
            boundary=[msg("1", "U1")],
            snapshot_uuid="p1",
            parent_snapshot_uuid=None,
            summary="S1",
            boundary_hash_override="bogus",
        ),
    )

    rebuilt = await syncer._rebuild_from_chain(
        conversation_uuid="conv",
        snapshot_uuid="p1",
        bot_author_id=BOT_ID,
        partial=_make_partial([msg("1", "U1")]),
        head_doc=None,
    )

    assert rebuilt.ancestor_summaries == []
    assert rebuilt.parent_snapshot_uuid is None
    assert rebuilt.raw_message_start_index == 0


@pytest.mark.asyncio
async def test_rebuild_from_chain_includes_only_summaries_up_to_matched_ancestor():
    """Three-generation chain where only the middle ancestor's boundary matches: ancestor_summaries includes
    the outer and middle, but NOT the inner (younger than the matched deepest-valid ancestor)."""
    syncer, _, search = make_syncer()
    p0_boundary = [msg("1", "U1"), bot_msg("2", "A1")]
    p1_boundary = p0_boundary + [msg("3", "U2"), bot_msg("4", "A2")]
    p2_boundary = p1_boundary + [msg("5", "U3"), bot_msg("6", "A3")]

    _seed_doc(search, _compacted_doc(boundary=p0_boundary, snapshot_uuid="p0", parent_snapshot_uuid=None, summary="S0"))
    _seed_doc(search, _compacted_doc(boundary=p1_boundary, snapshot_uuid="p1", parent_snapshot_uuid="p0", summary="S1"))
    # p2's boundary_hash points at a different set — doesn't match incoming.
    _seed_doc(
        search,
        _compacted_doc(
            boundary=p2_boundary,
            snapshot_uuid="p2",
            parent_snapshot_uuid="p1",
            summary="S2",
            boundary_hash_override="bogus",
        ),
    )

    incoming = p1_boundary + [msg("7", "U4")]
    rebuilt = await syncer._rebuild_from_chain(
        conversation_uuid="conv",
        snapshot_uuid="p2",
        bot_author_id=BOT_ID,
        partial=_make_partial(incoming),
        head_doc=None,
    )

    assert rebuilt.ancestor_summaries == ["S0", "S1"]
    assert rebuilt.parent_snapshot_uuid == "p1"


@pytest.mark.asyncio
async def test_rebuild_from_chain_uses_deepest_valid_compacted_ancestor():
    """When multiple ancestors have matching boundary_hashes, prefer the deepest (most-recent) — its
    raw_message_start_index is later, so less is replayed as raw window."""
    syncer, _, search = make_syncer()
    p0_boundary = [msg("1", "U1"), bot_msg("2", "A1")]
    p1_boundary = p0_boundary + [msg("3", "U2"), bot_msg("4", "A2")]

    _seed_doc(search, _compacted_doc(boundary=p0_boundary, snapshot_uuid="p0", parent_snapshot_uuid=None, summary="S0"))
    _seed_doc(search, _compacted_doc(boundary=p1_boundary, snapshot_uuid="p1", parent_snapshot_uuid="p0", summary="S1"))

    incoming = p1_boundary + [msg("5", "U3")]
    rebuilt = await syncer._rebuild_from_chain(
        conversation_uuid="conv",
        snapshot_uuid="p1",
        bot_author_id=BOT_ID,
        partial=_make_partial(incoming),
        head_doc=None,
    )

    # Deepest match (p1) is preferred — parent is p1, raw_start = p1's boundary count.
    assert rebuilt.parent_snapshot_uuid == "p1"
    assert rebuilt.raw_message_start_index == len(p1_boundary)


@pytest.mark.asyncio
async def test_rebuild_from_chain_restores_lifted_state_from_active_descendant():
    """Cold rebuild against a compacted ancestor must restore the descendant's `lifted_turn_items` +
    `lifted_anchor_source_id`. Without this, a Redis miss with the client anchored at the parent silently drops
    the file-tool live windows carried forward by compaction."""
    syncer, _, search = make_syncer()
    p0_boundary = [msg("1", "U1"), bot_msg("2", "A1")]
    p1_boundary = p0_boundary + [msg("3", "U2"), bot_msg("4", "A2")]

    # Compacted ancestor (matched_ancestor in the rebuild path).
    _seed_doc(
        search,
        _compacted_doc(boundary=p1_boundary, snapshot_uuid="p1", parent_snapshot_uuid=None, summary="S1"),
    )
    # Active descendant the compactor wrote with lifted state.
    from prokaryotes.conversation_v1.models import TurnItem
    lifted_items = [
        TurnItem(type="function_call", call_id="c1", name="file_tool"),
        TurnItem(type="function_call_output", call_id="c1", output="FILE ..."),
    ]
    _seed_doc(
        search,
        {
            "snapshot_uuid": "child",
            "conversation_uuid": "conv",
            "parent_snapshot_uuid": "p1",
            "bot_author_id": BOT_ID,
            "compaction_state": "committed",
            "is_compacted": False,
            "lifted_turn_items_json": json.dumps({"items": [i.model_dump() for i in lifted_items]}),
            "lifted_anchor_source_id": "4",
            "raw_message_start_index": len(p1_boundary),
            "messages_json": json.dumps({"messages": []}),
            "dt_modified": "2024-01-01T00:00:00",
        },
    )

    incoming = p1_boundary + [msg("5", "U3")]
    rebuilt = await syncer._rebuild_from_chain(
        conversation_uuid="conv",
        snapshot_uuid="p1",
        bot_author_id=BOT_ID,
        partial=_make_partial(incoming),
        head_doc=None,
    )

    assert rebuilt.parent_snapshot_uuid == "p1"
    assert len(rebuilt.lifted_turn_items) == 2
    assert rebuilt.lifted_anchor_source_id == "4"


@pytest.mark.asyncio
async def test_rebuild_from_chain_lifted_state_empty_when_no_descendant():
    """When the compacted ancestor has no active descendant (e.g. orphan parent), rebuilt lifted state is
    empty — `anchor=None iff lifted_turn_items==[]`."""
    syncer, _, search = make_syncer()
    boundary = [msg("1", "U1"), bot_msg("2", "A1")]
    _seed_doc(
        search,
        _compacted_doc(boundary=boundary, snapshot_uuid="p1", parent_snapshot_uuid=None, summary="S1"),
    )

    incoming = boundary + [msg("3", "U2")]
    rebuilt = await syncer._rebuild_from_chain(
        conversation_uuid="conv",
        snapshot_uuid="p1",
        bot_author_id=BOT_ID,
        partial=_make_partial(incoming),
        head_doc=None,
    )

    assert rebuilt.parent_snapshot_uuid == "p1"
    assert rebuilt.lifted_turn_items == []
    assert rebuilt.lifted_anchor_source_id is None


@pytest.mark.asyncio
async def test_rebuild_from_chain_returns_fresh_when_no_snapshot_uuid():
    """A fresh conversation (no client-supplied snapshot_uuid) returns a fresh Conversation with no
    ancestors."""
    syncer, _, _ = make_syncer()

    rebuilt = await syncer._rebuild_from_chain(
        conversation_uuid="conv",
        snapshot_uuid=None,
        bot_author_id=BOT_ID,
        partial=_make_partial([msg("1", "U1")]),
        head_doc=None,
    )

    assert rebuilt.ancestor_summaries == []
    assert rebuilt.parent_snapshot_uuid is None
    assert rebuilt.messages == []


# -----------------------------------------------------------------------------
# stream_and_finalize handshake + compaction_pending
# -----------------------------------------------------------------------------


class _StubHarness(HarnessBase):
    """Concrete HarnessBase for stream_and_finalize tests. Stubs out finalize_turn so we don't need to wire ES
    persistence."""

    def __init__(self, redis: FakeRedis, search: FakeSearchClient):
        self._redis_client = redis
        self._search_client = search
        self.background_tasks = set()
        self._finalize_calls: list[dict] = []

    @property
    def conversation_cache_ex(self) -> int:
        return 3600

    async def finalize_turn(self, *, conversation, bot_message_source_id, bot_message_content, turn_items):
        self._finalize_calls.append(
            {
                "snapshot_uuid": conversation.snapshot_uuid,
                "bot_message_source_id": bot_message_source_id,
                "bot_message_content": bot_message_content,
                "turn_items": list(turn_items),
            }
        )

    def background_and_forget(self, coro):
        self.background_tasks.add(coro)
        coro.close()  # don't schedule — these tests only check emission ordering


def _make_sync_result(snapshot_uuid: str = "snap") -> SyncResult:
    from tests.unit_tests._builders import conversation

    return SyncResult(
        conversation=conversation(msg("1", "U1"), snapshot_uuid=snapshot_uuid),
        source_id_assignments=[SourceIdAssignment(client_index=0, source_id="1")],
        is_new_branch=False,
    )


@pytest.mark.asyncio
async def test_stream_and_finalize_emits_handshake_first_and_compaction_pending_last():
    """First event = handshake carrying snapshot_uuid and source_id_assignments. Last event before completion
    = compaction_pending (when lock acquired)."""
    redis = FakeRedis()
    search = FakeSearchClient()
    harness = _StubHarness(redis, search)
    sync_result = _make_sync_result()

    async def factory(ctx):
        ctx.final_assistant_text.append("Done.")
        yield '{"text_delta": "Done."}\n'

    async def compact_fn(_snapshot, _prep):
        return "Summary"

    chunks: list[str] = []
    async for chunk in harness.stream_and_finalize(
        sync_result=sync_result,
        bot_message_source_id_provider=lambda c: "2.000000",
        response_generator_factory=factory,
        compact_fn=compact_fn,
        pending_compaction=[True],
    ):
        chunks.append(chunk)

    # First event: handshake.
    first = json.loads(chunks[0])
    assert first["snapshot_uuid"] == "snap"
    assert first["source_id_assignments"] == [{"client_index": 0, "source_id": "1"}]
    # Last event: compaction_pending.
    last = json.loads(chunks[-1])
    assert last == {"compaction_pending": True}
    # bot_message arrives before compaction_pending.
    parsed = [json.loads(c) for c in chunks]
    bot_idx = next(i for i, p in enumerate(parsed) if "bot_message" in p)
    pending_idx = next(i for i, p in enumerate(parsed) if "compaction_pending" in p)
    assert bot_idx < pending_idx


@pytest.mark.asyncio
async def test_stream_and_finalize_skips_duplicate_compaction_when_lock_held():
    """If `compaction_lock:` already exists (`nx=True` fails), no compaction_pending event and no background
    compaction kicked off."""
    redis = FakeRedis()
    await redis.set("compaction_lock:c-1", "1")  # pre-acquired lock
    search = FakeSearchClient()
    harness = _StubHarness(redis, search)
    sync_result = _make_sync_result()

    async def factory(ctx):
        ctx.final_assistant_text.append("Done.")
        yield '{"text_delta": "Done."}\n'

    async def compact_fn(_snapshot, _prep):
        return "Summary"

    chunks: list[str] = []
    async for chunk in harness.stream_and_finalize(
        sync_result=sync_result,
        bot_message_source_id_provider=lambda c: "2.000000",
        response_generator_factory=factory,
        compact_fn=compact_fn,
        pending_compaction=[True],
    ):
        chunks.append(chunk)

    parsed = [json.loads(c) for c in chunks]
    assert not any("compaction_pending" in p for p in parsed)
    assert len(harness.background_tasks) == 0


@pytest.mark.asyncio
async def test_stream_and_finalize_omits_bot_message_when_no_final_text():
    """If the LLM stream aborts without producing final assistant text, no bot_message event is emitted and no
    finalize_turn happens (the client must not create an assistant node)."""
    redis = FakeRedis()
    search = FakeSearchClient()
    harness = _StubHarness(redis, search)
    sync_result = _make_sync_result()

    async def factory(ctx):
        # No final_assistant_text appended.
        yield '{"context_pct": 50}\n'

    chunks: list[str] = []
    async for chunk in harness.stream_and_finalize(
        sync_result=sync_result,
        bot_message_source_id_provider=lambda c: "2.000000",
        response_generator_factory=factory,
        compact_fn=None,
        pending_compaction=None,
    ):
        chunks.append(chunk)

    parsed = [json.loads(c) for c in chunks]
    assert not any("bot_message" in p for p in parsed)
    assert harness._finalize_calls == []


@pytest.mark.asyncio
async def test_stream_and_finalize_resync_closes_stream_after_handshake():
    """On resync, the handshake carries unacknowledged_bot_messages and the stream closes without invoking the
    LLM."""
    from prokaryotes.context_v1.conversation_sync import UnacknowledgedBotMessage
    from tests.unit_tests._builders import conversation

    redis = FakeRedis()
    search = FakeSearchClient()
    harness = _StubHarness(redis, search)

    sync_result = SyncResult(
        conversation=conversation(msg("1", "U1"), snapshot_uuid="snap"),
        source_id_assignments=[],
        is_new_branch=False,
        resync=True,
        unacknowledged_bot_messages=[
            UnacknowledgedBotMessage(source_id="2", content="A1", parent_source_id="1"),
        ],
    )

    factory_called = [False]

    async def factory(ctx):
        factory_called[0] = True
        yield ""

    chunks: list[str] = []
    async for chunk in harness.stream_and_finalize(
        sync_result=sync_result,
        bot_message_source_id_provider=lambda c: "X",
        response_generator_factory=factory,
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    handshake = json.loads(chunks[0])
    assert handshake["snapshot_uuid"] == "snap"
    assert handshake["unacknowledged_bot_messages"] == [
        {"source_id": "2", "content": "A1", "parent_source_id": "1"},
    ]
    # Factory never called on resync.
    assert factory_called[0] is False
