"""Tests for the DAG-scoped assistant-message guardrail.

`validate_assistant_messages` rejects three classes of corruption:
- assistant entry with no `source_id` (web clients can't author bot messages)
- assistant `source_id` not in any `ConversationMessage` for the DAG
- assistant `source_id` known but with mismatched content

`load_assistant_index` walks every conversation doc for the `conversation_uuid`, including compacted ancestors
(whose `messages_json` is preserved). The cache in Redis (`assistant_index:{conversation_uuid}`) is refreshed on
every bot commit via `refresh_assistant_index_with`.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from prokaryotes.api_v1.models import IncomingMessage
from prokaryotes.context_v1.conversation_sync import AssistantMessageGuardrailError
from tests.unit_tests._builders import bot_msg, conversation, msg
from tests.unit_tests._fakes import make_syncer


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class TestLoadAssistantIndex:
    @pytest.mark.asyncio
    async def test_empty_dag_returns_empty_index(self):
        syncer, _redis, _search = make_syncer()
        index = await syncer.load_assistant_index("c-fresh")
        assert index == {}

    @pytest.mark.asyncio
    async def test_indexes_bot_messages_across_dag(self):
        syncer, _redis, search = make_syncer()
        # Two snapshots in the DAG: a compacted parent and a non-compacted child.
        parent = conversation(
            msg("1.000001", "u1"),
            bot_msg("1.000002", "b1"),
            msg("1.000003", "u2"),
            bot_msg("1.000004", "b2"),
            conversation_uuid="c-1",
            snapshot_uuid="s-parent",
        )
        search.store_conversation_doc(parent, is_compacted=True, summary="<sum>")
        child = conversation(
            msg("1.000003", "u2"),
            bot_msg("1.000004", "b2"),
            bot_msg("1.000005", "b3"),
            conversation_uuid="c-1",
            snapshot_uuid="s-child",
            parent_snapshot_uuid="s-parent",
            ancestor_summaries=["<sum>"],
            raw_message_start_index=2,
        )
        search.store_conversation_doc(child)

        index = await syncer.load_assistant_index("c-1")
        # Every bot reachable in the DAG; b2 appears on both snapshots (same hash).
        assert set(index.keys()) == {"1.000002", "1.000004", "1.000005"}
        assert index["1.000002"] == _hash("b1")
        assert index["1.000004"] == _hash("b2")
        assert index["1.000005"] == _hash("b3")

    @pytest.mark.asyncio
    async def test_excludes_user_and_tombstoned_messages(self):
        syncer, _redis, search = make_syncer()
        snap = conversation(
            msg("1.000001", "u1"),
            bot_msg("1.000002", "b1"),
            msg("1.000003", "u2"),
            bot_msg("1.000004", "deleted bot"),
            conversation_uuid="c-1",
            snapshot_uuid="s-snap",
        )
        snap.message_by_source_id("1.000004").deleted = True
        search.store_conversation_doc(snap)

        index = await syncer.load_assistant_index("c-1")
        assert set(index.keys()) == {"1.000002"}

    @pytest.mark.asyncio
    async def test_caches_in_redis(self):
        syncer, redis, search = make_syncer()
        snap = conversation(
            msg("1.000001", "u1"),
            bot_msg("1.000002", "b1"),
            conversation_uuid="c-1",
            snapshot_uuid="s-snap",
        )
        search.store_conversation_doc(snap)
        await syncer.load_assistant_index("c-1")

        cached = await redis.get("assistant_index:c-1")
        assert cached is not None
        decoded = json.loads(cached.decode("utf-8"))
        assert decoded == {"1.000002": _hash("b1")}

    @pytest.mark.asyncio
    async def test_redis_hit_short_circuits_es(self):
        """A primed Redis cache means we don't have to walk ES on every call."""
        syncer, redis, search = make_syncer()
        # Prime the cache directly; do NOT seed ES. If the cache hit fails to short-circuit, the index would
        # come back empty.
        await redis.set(
            "assistant_index:c-1",
            json.dumps({"1.000002": _hash("b1")}),
        )
        assert search.conversations == {}

        index = await syncer.load_assistant_index("c-1")
        assert index == {"1.000002": _hash("b1")}


class TestRefreshAssistantIndexWith:
    @pytest.mark.asyncio
    async def test_appends_to_index_and_updates_cache(self):
        syncer, redis, search = make_syncer()
        snap = conversation(
            msg("1.000001", "u1"),
            bot_msg("1.000002", "b1"),
            conversation_uuid="c-1",
            snapshot_uuid="s-snap",
        )
        search.store_conversation_doc(snap)
        # Initial cache load.
        await syncer.load_assistant_index("c-1")

        # Refresh with a new bot message — typically called from finalize_turn.
        await syncer.refresh_assistant_index_with("c-1", "1.000005", "new bot reply")

        # Cache now contains both entries.
        cached = json.loads((await redis.get("assistant_index:c-1")).decode("utf-8"))
        assert cached == {
            "1.000002": _hash("b1"),
            "1.000005": _hash("new bot reply"),
        }

    @pytest.mark.asyncio
    async def test_refresh_overwrites_existing_entry(self):
        """Repeated refresh for the same source_id replaces the content hash."""
        syncer, redis, _search = make_syncer()
        await redis.set(
            "assistant_index:c-1",
            json.dumps({"1.000002": _hash("original")}),
        )
        await syncer.refresh_assistant_index_with("c-1", "1.000002", "different")
        cached = json.loads((await redis.get("assistant_index:c-1")).decode("utf-8"))
        assert cached == {"1.000002": _hash("different")}


class TestValidateAssistantMessages:
    @pytest.mark.asyncio
    async def test_pure_user_payload_passes(self):
        """No assistant entries → no check, no ES round-trip."""
        syncer, _redis, _search = make_syncer()
        await syncer.validate_assistant_messages(
            "c-fresh",
            [IncomingMessage(role="user", content="hi")],
        )

    @pytest.mark.asyncio
    async def test_assistant_without_source_id_rejected(self):
        syncer, _redis, _search = make_syncer()
        with pytest.raises(AssistantMessageGuardrailError, match="server-assigned source_id"):
            await syncer.validate_assistant_messages(
                "c-1",
                [
                    IncomingMessage(role="user", content="u1"),
                    IncomingMessage(role="assistant", content="fake bot"),
                ],
            )

    @pytest.mark.asyncio
    async def test_unknown_assistant_source_id_rejected(self):
        syncer, _redis, search = make_syncer()
        snap = conversation(
            msg("1.000001", "u1"),
            bot_msg("1.000002", "b1"),
            conversation_uuid="c-1",
            snapshot_uuid="s-snap",
        )
        search.store_conversation_doc(snap)
        with pytest.raises(AssistantMessageGuardrailError, match="Unknown assistant source_id"):
            await syncer.validate_assistant_messages(
                "c-1",
                [
                    IncomingMessage(role="user", content="u1", source_id="1.000001"),
                    IncomingMessage(
                        role="assistant",
                        content="fabricated",
                        source_id="1.999999",
                    ),
                ],
            )

    @pytest.mark.asyncio
    async def test_known_source_id_with_different_content_rejected(self):
        syncer, _redis, search = make_syncer()
        snap = conversation(
            msg("1.000001", "u1"),
            bot_msg("1.000002", "original bot text"),
            conversation_uuid="c-1",
            snapshot_uuid="s-snap",
        )
        search.store_conversation_doc(snap)
        with pytest.raises(AssistantMessageGuardrailError, match="content mismatch"):
            await syncer.validate_assistant_messages(
                "c-1",
                [
                    IncomingMessage(role="user", content="u1", source_id="1.000001"),
                    IncomingMessage(
                        role="assistant",
                        content="rewritten bot text",
                        source_id="1.000002",
                    ),
                ],
            )

    @pytest.mark.asyncio
    async def test_valid_echoed_assistant_accepted(self):
        """Client echoing a known bot message with matching content passes."""
        syncer, _redis, search = make_syncer()
        snap = conversation(
            msg("1.000001", "u1"),
            bot_msg("1.000002", "real bot text"),
            conversation_uuid="c-1",
            snapshot_uuid="s-snap",
        )
        search.store_conversation_doc(snap)
        # Should not raise.
        await syncer.validate_assistant_messages(
            "c-1",
            [
                IncomingMessage(role="user", content="u1", source_id="1.000001"),
                IncomingMessage(
                    role="assistant",
                    content="real bot text",
                    source_id="1.000002",
                ),
                IncomingMessage(role="user", content="follow-up"),
            ],
        )

    @pytest.mark.asyncio
    async def test_case_b_retry_with_compacted_ancestor_assistant_accepted(self):
        """Case B path: client echoes assistant `source_id`s that live only in the compacted parent's retained
        `messages_json`. The DAG-scoped lookup hits the parent, validation passes."""
        syncer, _redis, search = make_syncer()
        # Compacted parent has the bot; child snapshot has only the post-tail.
        parent = conversation(
            msg("1.000001", "u1"),
            bot_msg("1.000002", "compacted-away bot"),
            msg("1.000003", "u2"),
            bot_msg("1.000004", "still-visible bot"),
            conversation_uuid="c-1",
            snapshot_uuid="s-parent",
        )
        search.store_conversation_doc(parent, is_compacted=True, summary="<sum>")
        child = conversation(
            msg("1.000003", "u2"),
            bot_msg("1.000004", "still-visible bot"),
            conversation_uuid="c-1",
            snapshot_uuid="s-child",
            parent_snapshot_uuid="s-parent",
            ancestor_summaries=["<sum>"],
            raw_message_start_index=2,
        )
        search.store_conversation_doc(child)

        # Client edits something that requires referencing the compacted bot. Validation must accept
        # "1.000002" because the parent's messages_json still has it.
        await syncer.validate_assistant_messages(
            "c-1",
            [
                IncomingMessage(role="user", content="u1", source_id="1.000001"),
                IncomingMessage(
                    role="assistant",
                    content="compacted-away bot",
                    source_id="1.000002",
                ),
                IncomingMessage(role="user", content="new question"),
            ],
        )
