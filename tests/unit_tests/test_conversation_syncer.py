"""End-to-end tests for `ConversationSyncer.sync_conversation`.

These run the full pipeline (load → split → resync → assign → reconcile → apply → persist) against the in-memory
fakes. They cover Issues 1, 3, and the ordering invariants between the compacted-prefix split and resync detection.
"""

from __future__ import annotations

import pytest

from prokaryotes.api_v1.models import IncomingMessage
from tests.unit_tests._builders import BOT_ID, bot_msg, conversation, msg
from tests.unit_tests._fakes import make_syncer


def _store_redis_snapshot(redis_client, conversation_obj):
    """Seed Redis so `_load_stored` hits Tier 1 and returns this snapshot."""
    redis_client._store[f"conversation:{conversation_obj.conversation_uuid}"] = (
        conversation_obj.model_dump_json().encode("utf-8")
    )


def _setup_compacted_state(search, redis, *, parent_messages, child_messages, raw_start, child_uuid="s-child"):
    parent = conversation(
        *parent_messages,
        snapshot_uuid="s-parent",
        ancestor_summaries=[],
    )
    search.store_conversation_doc(parent, is_compacted=True, summary="<sum>")
    child = conversation(
        *child_messages,
        snapshot_uuid=child_uuid,
        parent_snapshot_uuid="s-parent",
        ancestor_summaries=["<sum>"],
        raw_message_start_index=raw_start,
    )
    search.store_conversation_doc(child)
    _store_redis_snapshot(redis, child)
    return child


class TestPostCompactionAppendUsesRawSuffix:
    """Issue 1's blocker scenario: client sends full echoed history after
    compaction; sync drops the compacted prefix and appends only the new user message to the child snapshot."""

    @pytest.mark.asyncio
    async def test_appends_only_new_entry(self):
        syncer, redis, search = make_syncer()
        parent_messages = [
            msg("1", "u1"),
            bot_msg("2", "b1"),
            msg("3", "u2"),
            bot_msg("4", "b2"),
        ]
        child_messages = parent_messages[2:]  # [m3, m4]
        _setup_compacted_state(
            search,
            redis,
            parent_messages=parent_messages,
            child_messages=child_messages,
            raw_start=2,
        )

        incoming = [
            IncomingMessage(role="user", content="u1", source_id="1"),
            IncomingMessage(role="assistant", content="b1", source_id="2"),
            IncomingMessage(role="user", content="u2", source_id="3"),
            IncomingMessage(role="assistant", content="b2", source_id="4"),
            IncomingMessage(role="user", content="follow-up"),  # bare — needs assignment
        ]

        result = await syncer.sync_conversation(
            conversation_uuid="c-1",
            snapshot_uuid="s-child",
            bot_author_id=BOT_ID,
            incoming=incoming,
            session_user_id="u-alice",
            session_display_name="Alice",
        )

        # Only the raw window + new entry in child.messages — m1, m2 stay summarized into ancestor_summaries.
        assigned_sid = result.source_id_assignments[0].source_id
        assert [m.source_id for m in result.conversation.messages] == ["3", "4", assigned_sid]
        assert [m.content for m in result.conversation.messages] == ["u2", "b2", "follow-up"]
        assert result.conversation.ancestor_summaries == ["<sum>"]
        # raw_message_start_index preserved — same compacted snapshot.
        assert result.conversation.raw_message_start_index == 2
        # Child snapshot extended in place — no new branch.
        assert result.is_new_branch is False
        assert result.conversation.snapshot_uuid == "s-child"
        # Exactly one source_id assigned (the bare follow-up).
        assert len(result.source_id_assignments) == 1
        assert result.source_id_assignments[0].client_index == 4

    @pytest.mark.asyncio
    async def test_no_duplicate_messages_after_sync(self):
        """Regression: before Issue 1 was fixed, the compacted-prefix entries
        would get re-appended to stored.messages, producing duplicates after re-sort. Verify the duplicate-injection
        doesn't happen."""
        syncer, redis, search = make_syncer()
        _setup_compacted_state(
            search,
            redis,
            parent_messages=[
                msg("1", "u1"),
                bot_msg("2", "b1"),
                msg("3", "u2"),
                bot_msg("4", "b2"),
            ],
            child_messages=[msg("3", "u2"), bot_msg("4", "b2")],
            raw_start=2,
        )

        incoming = [
            IncomingMessage(role="user", content="u1", source_id="1"),
            IncomingMessage(role="assistant", content="b1", source_id="2"),
            IncomingMessage(role="user", content="u2", source_id="3"),
            IncomingMessage(role="assistant", content="b2", source_id="4"),
            IncomingMessage(role="user", content="next"),
        ]
        result = await syncer.sync_conversation(
            conversation_uuid="c-1",
            snapshot_uuid="s-child",
            bot_author_id=BOT_ID,
            incoming=incoming,
            session_user_id="u-alice",
            session_display_name="Alice",
        )

        seen_source_ids = [m.source_id for m in result.conversation.messages]
        assert seen_source_ids == sorted(set(seen_source_ids))  # no dups, sorted
        assert "1" not in seen_source_ids  # m1 stays summarized
        assert "2" not in seen_source_ids  # m2 stays summarized


class TestSplitRunsBeforeResyncDetection:
    """Without the split running first, compacted-prefix entries pollute the
    resync detector's `has_new_incoming` heuristic and the detector misfires. With the split, the detector sees only
    the raw suffix."""

    @pytest.mark.asyncio
    async def test_compacted_prefix_does_not_trigger_false_resync(self):
        """Stored snapshot has compacted prefix + raw tail with NO trailing
        un-acked bot. Client echoes prefix + raw tail + new user message. With the split, the detector sees only the
        raw suffix, finds no trailing bot, and lets the flow continue."""
        syncer, redis, search = make_syncer()
        _setup_compacted_state(
            search,
            redis,
            parent_messages=[
                msg("1", "u1"),
                bot_msg("2", "b1"),
                msg("3", "u2"),
                bot_msg("4", "b2"),
            ],
            child_messages=[msg("3", "u2"), bot_msg("4", "b2")],
            raw_start=2,
        )

        incoming = [
            IncomingMessage(role="user", content="u1", source_id="1"),
            IncomingMessage(role="assistant", content="b1", source_id="2"),
            IncomingMessage(role="user", content="u2", source_id="3"),
            IncomingMessage(role="assistant", content="b2", source_id="4"),
            IncomingMessage(role="user", content="follow-up"),
        ]
        result = await syncer.sync_conversation(
            conversation_uuid="c-1",
            snapshot_uuid="s-child",
            bot_author_id=BOT_ID,
            incoming=incoming,
            session_user_id="u-alice",
            session_display_name="Alice",
        )
        assert result.resync is False
        assert result.unacknowledged_bot_messages == []

    @pytest.mark.asyncio
    async def test_trailing_unacked_bot_after_raw_window_still_detected(self):
        """Resync detection still fires on a real trailing un-acked bot,
        even when a compacted prefix is also present."""
        syncer, redis, search = make_syncer()
        parent_messages = [
            msg("1", "u1"),
            bot_msg("2", "b1"),
            msg("3", "u2"),
            bot_msg("4", "b2"),
        ]
        # Child has [u2, b2, b2-followup] — the followup is the trailing un-acked bot.
        child_messages = [msg("3", "u2"), bot_msg("4", "b2"), bot_msg("5", "b2-followup")]
        _setup_compacted_state(
            search,
            redis,
            parent_messages=parent_messages,
            child_messages=child_messages,
            raw_start=2,
        )

        # Client only saw up through "b2" (m4) and is sending a new user message.
        incoming = [
            IncomingMessage(role="user", content="u1", source_id="1"),
            IncomingMessage(role="assistant", content="b1", source_id="2"),
            IncomingMessage(role="user", content="u2", source_id="3"),
            IncomingMessage(role="assistant", content="b2", source_id="4"),
            IncomingMessage(role="user", content="next"),
        ]
        result = await syncer.sync_conversation(
            conversation_uuid="c-1",
            snapshot_uuid="s-child",
            bot_author_id=BOT_ID,
            incoming=incoming,
            session_user_id="u-alice",
            session_display_name="Alice",
        )
        assert result.resync is True
        assert len(result.unacknowledged_bot_messages) == 1
        assert result.unacknowledged_bot_messages[0].source_id == "5"
        assert result.unacknowledged_bot_messages[0].content == "b2-followup"
        # Resync deferred source_id assignment.
        assert result.source_id_assignments == []


class TestCaseBHelperAssignsSourceIds:
    """Case B: compacted prefix mismatches — discard stored, build fresh
    Conversation from full incoming. Bare entries get source_ids assigned; handshake's source_id_assignments must
    carry the right client_index.

    The realistic Case B scenario is "user edits a message that was compacted
    away": the edit creates a new client node with no source_id, the active path is shorter than the compacted
    prefix, length mismatch routes to Case B."""

    @pytest.mark.asyncio
    async def test_edit_of_compacted_message_creates_fresh_conversation(self):
        syncer, redis, search = make_syncer()
        _setup_compacted_state(
            search,
            redis,
            parent_messages=[
                msg("1717000001.000000", "u1"),
                bot_msg("1717000002.000000", "b1"),
                msg("1717000003.000000", "u2"),
                bot_msg("1717000004.000000", "b2"),
            ],
            child_messages=[
                msg("1717000003.000000", "u2"),
                bot_msg("1717000004.000000", "b2"),
            ],
            raw_start=2,
        )

        # User edits u1: active path is [u1', new-typed-message]. Both client nodes have no source_id (edit + fresh
        # type), so the prefix split fails and routes to Case B.
        incoming = [
            IncomingMessage(role="user", content="u1 edited"),
            IncomingMessage(role="user", content="follow-up after edit"),
        ]
        result = await syncer.sync_conversation(
            conversation_uuid="c-1",
            snapshot_uuid="s-child",
            bot_author_id=BOT_ID,
            incoming=incoming,
            session_user_id="u-alice",
            session_display_name="Alice",
        )

        # Fresh conversation — no inherited compacted state.
        assert result.conversation.ancestor_summaries == []
        assert result.conversation.working_file_windows == []
        assert result.conversation.raw_message_start_index == 0
        # New snapshot_uuid != client's `s-child`.
        assert result.is_new_branch is True
        assert result.conversation.snapshot_uuid != "s-child"
        # Both bare entries got source_ids assigned in submission order.
        assert len(result.source_id_assignments) == 2
        assert [a.client_index for a in result.source_id_assignments] == [0, 1]
        # Assignments are monotonic (microsecond-bumped if collision).
        first, second = result.source_id_assignments
        assert second.source_id > first.source_id
        # Conversation has the two new messages in submission order.
        assert [m.content for m in result.conversation.messages] == [
            "u1 edited",
            "follow-up after edit",
        ]

    @pytest.mark.asyncio
    async def test_source_id_mismatch_in_prefix_routes_to_case_b(self):
        """The same-content / different-source_id case: client sent enough
        entries to potentially match the compacted prefix in length, but at least one source_id is fabricated. Split
        fails on the source_id check and routes to Case B."""
        syncer, redis, search = make_syncer()
        _setup_compacted_state(
            search,
            redis,
            parent_messages=[
                msg("1717000001.000000", "u1"),
                bot_msg("1717000002.000000", "b1"),
                msg("1717000003.000000", "u2"),
                bot_msg("1717000004.000000", "b2"),
            ],
            child_messages=[
                msg("1717000003.000000", "u2"),
                bot_msg("1717000004.000000", "b2"),
            ],
            raw_start=2,
        )
        # Client echoed m1, m2 but with a fabricated source_id for m1.
        incoming = [
            IncomingMessage(role="user", content="u1", source_id="1717000001.999999"),
            IncomingMessage(role="assistant", content="b1", source_id="1717000002.000000"),
            IncomingMessage(role="user", content="u2", source_id="1717000003.000000"),
            IncomingMessage(role="assistant", content="b2", source_id="1717000004.000000"),
        ]
        result = await syncer.sync_conversation(
            conversation_uuid="c-1",
            snapshot_uuid="s-child",
            bot_author_id=BOT_ID,
            incoming=incoming,
            session_user_id="u-alice",
            session_display_name="Alice",
        )
        # Routes to Case B because of the fabricated source_id.
        assert result.is_new_branch is True
        assert result.conversation.ancestor_summaries == []
        assert result.conversation.raw_message_start_index == 0
