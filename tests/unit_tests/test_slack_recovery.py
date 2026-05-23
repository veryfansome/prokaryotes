"""`SlackHarness._load_slack_stored` cold-recovery paths.

Slack has no client echoing a `snapshot_uuid` back, so the harness recovers the active snapshot itself: Redis
fast path → `find_latest_active_snapshot_uuid` exact-load → fresh `Conversation` on a total miss. A snapshot
that is compacted or pending is never selected as the head.
"""

from __future__ import annotations

import pytest

from prokaryotes.conversation_v1.models import Conversation, ConversationMessage
from prokaryotes.harness_v1.slack import SlackHarness
from tests.unit_tests._slack_fakes import FakeRedis, FakeSearchClient

BOT = "U_BOT"
CONV_UUID = "c-slack-1"


class _TestHarness(SlackHarness):
    """`SlackHarness` with fakes injected, bypassing `__init__`'s LLM-client construction."""

    def __init__(self) -> None:
        # Skip SlackHarness.__init__ — it builds a real LLM client. Wire only what `_load_slack_stored` needs.
        self._redis_client = FakeRedis()
        self._search_client = FakeSearchClient()
        self._conversation_cache_ex = 60 * 60 * 24 * 7
        self.bot_user_id = BOT


def _conv(snapshot_uuid: str, *, content: str = "hello", parent: str | None = None) -> Conversation:
    return Conversation(
        conversation_uuid=CONV_UUID,
        snapshot_uuid=snapshot_uuid,
        parent_snapshot_uuid=parent,
        bot_author_id=BOT,
        messages=[ConversationMessage(source_id="100.0", author_id="U_ALICE", content=content)],
    )


@pytest.fixture
def harness() -> _TestHarness:
    return _TestHarness()


@pytest.mark.asyncio
async def test_redis_cached_returns_without_es_roundtrip(harness: _TestHarness):
    """A cached `Conversation` is returned straight from Redis with no ES query."""
    cached = _conv("s-cached", content="from redis")
    await harness._redis_client.set(f"conversation:{CONV_UUID}", cached.model_dump_json())

    result = await harness._load_slack_stored(CONV_UUID)

    assert result.snapshot_uuid == "s-cached"
    assert result.messages[0].content == "from redis"
    # ES was never consulted.
    assert harness._search_client.conversations == {}


@pytest.mark.asyncio
async def test_redis_cold_es_active_snapshot_loads_it(harness: _TestHarness):
    """Redis cold, ES has an active snapshot → `find_latest_active_snapshot_uuid` seeds the exact load."""
    stored = _conv("s-active", content="from elasticsearch")
    await harness._search_client.put_conversation(stored)

    result = await harness._load_slack_stored(CONV_UUID)

    assert result.snapshot_uuid == "s-active"
    assert result.messages[0].content == "from elasticsearch"


@pytest.mark.asyncio
async def test_redis_cold_es_empty_returns_fresh_conversation(harness: _TestHarness):
    """Redis cold, ES empty → a fresh `Conversation` keyed to the harness's `bot_user_id`."""
    result = await harness._load_slack_stored(CONV_UUID)

    assert result.conversation_uuid == CONV_UUID
    assert result.bot_author_id == BOT
    assert result.messages == []


@pytest.mark.asyncio
async def test_redis_cold_only_compacted_snapshot_returns_fresh(harness: _TestHarness):
    """When ES holds only a compacted snapshot, `find_latest_active_snapshot_uuid` returns `None` and the
    harness starts fresh rather than re-attaching to a compacted head."""
    harness._search_client.store_conversation_doc(_conv("s-compacted"), is_compacted=True)

    result = await harness._load_slack_stored(CONV_UUID)

    assert result.messages == []
    assert result.bot_author_id == BOT


@pytest.mark.asyncio
async def test_compaction_window_redis_loss_recovers_parent(harness: _TestHarness):
    """Redis lost during compaction's between-Redis-and-child-commit window: the committed parent snapshot is
    still active (`is_compacted=False`) while the child is `pending`. The query returns the parent; the
    abandoned pending child is ignored."""
    parent = _conv("s-parent", content="parent active snapshot")
    pending_child = _conv("s-pending-child", content="abandoned child", parent="s-parent")

    # Parent: committed, not yet marked compacted (the compactor never reached the swap).
    await harness._search_client.put_conversation(parent)
    # Pending child: a pending compaction attempt that was abandoned.
    harness._search_client.store_conversation_doc(pending_child, compaction_state="pending")

    result = await harness._load_slack_stored(CONV_UUID)

    assert result.snapshot_uuid == "s-parent"
    assert result.messages[0].content == "parent active snapshot"


@pytest.mark.asyncio
async def test_corrupt_redis_cache_falls_back_to_es(harness: _TestHarness):
    """A corrupt cached value does not crash recovery — it falls back to the ES query."""
    await harness._redis_client.set(f"conversation:{CONV_UUID}", "{not json")
    stored = _conv("s-active", content="recovered from es")
    await harness._search_client.put_conversation(stored)

    result = await harness._load_slack_stored(CONV_UUID)

    assert result.snapshot_uuid == "s-active"
    assert result.messages[0].content == "recovered from es"
