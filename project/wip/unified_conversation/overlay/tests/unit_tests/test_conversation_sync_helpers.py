"""Unit tests for `conversation_sync` pure helpers.

The reconcile-tier (Redis/ES/chain rebuild) tests live with the full syncer
suite once we have a fake SearchClient + Redis. These exercise just the
helpers that don't need infra.
"""

from __future__ import annotations

from prokaryotes.context_v1.conversation_sync import (
    _bump_source_id,
    _conversation_can_follow_client,
    _detect_unacknowledged_bot_messages,
    _format_source_id,
    _PartialMessage,
)
from tests.unit_tests._builders import bot_msg, conversation, msg


def _partial(content: str, source_id: str | None, *, author_id: str = "u-alice", client_index: int = 0):
    return _PartialMessage(
        author_id=author_id,
        content=content,
        client_index=client_index,
        source_id=source_id,
    )


class TestFormatSourceId:
    def test_seconds_microseconds_format(self):
        out = _format_source_id(1717000000.123456)
        assert out == "1717000000.123456"

    def test_zero_pads_microseconds(self):
        out = _format_source_id(1717000000.0)
        assert out == "1717000000.000000"

    def test_lex_sort_equals_chronological_for_realistic_timestamps(self):
        """The format relies on fixed-width seconds (10-digit unix timestamps in this
        era). Under that constraint, lexicographic sort equals chronological sort —
        which is what makes `source_id` viable as both identity and ordering."""
        a = _format_source_id(1717000000.123456)
        b = _format_source_id(1717000001.000000)
        c = _format_source_id(1717000999.999999)
        assert a < b < c


class TestBumpSourceId:
    def test_increments_microseconds(self):
        assert _bump_source_id("1717000000.000000") == "1717000000.000001"
        assert _bump_source_id("1717000000.123456") == "1717000000.123457"

    def test_carries_seconds_at_microsecond_overflow(self):
        assert _bump_source_id("1717000000.999999") == "1717000001.000000"

    def test_invalid_input_falls_back(self):
        # Doesn't crash; produces a fresh now-time source_id.
        out = _bump_source_id("not-a-source-id")
        assert "." in out


class TestConversationCanFollowClient:
    def test_none_client_uuid_allows_any_cached(self):
        c = conversation(msg("1", "hi"), snapshot_uuid="s-A")
        assert _conversation_can_follow_client(c, None)

    def test_matching_snapshot_uuid(self):
        c = conversation(msg("1", "hi"), snapshot_uuid="s-A")
        assert _conversation_can_follow_client(c, "s-A")

    def test_matching_parent_snapshot_uuid(self):
        """A cached child snapshot can serve a client whose own pointer is the parent —
        the relabel hasn't reached the client yet."""
        c = conversation(msg("1", "hi"), snapshot_uuid="s-child", parent_snapshot_uuid="s-parent")
        assert _conversation_can_follow_client(c, "s-parent")

    def test_unrelated_snapshot_uuid_rejected(self):
        c = conversation(msg("1", "hi"), snapshot_uuid="s-A", parent_snapshot_uuid="s-P")
        assert not _conversation_can_follow_client(c, "s-other")


class TestDetectUnacknowledgedBotMessages:
    def test_empty_when_stored_is_empty(self):
        stored = conversation()
        assert _detect_unacknowledged_bot_messages(stored, [_partial("hi", None)]) == []

    def test_empty_when_no_shared_source_id(self):
        """No prefix in common → not a resync; either fresh conversation or full divergence."""
        stored = conversation(msg("1", "hi"), bot_msg("2", "reply"))
        partial = [_partial("new", None)]
        assert _detect_unacknowledged_bot_messages(stored, partial) == []

    def test_detects_single_trailing_bot(self):
        """The post-commit stream-loss case: stored has trailing bot the client never saw."""
        stored = conversation(
            msg("1", "u1"),
            bot_msg("2", "b1"),
            msg("3", "u2"),
            bot_msg("4", "b2 (just committed)"),
        )
        # Client retries with the user history up through u2 and a new message
        partial = [
            _partial("u1", "1"),
            _partial("u2", "3"),
            _partial("new question", None),
        ]
        result = _detect_unacknowledged_bot_messages(stored, partial)
        assert len(result) == 1
        assert result[0].source_id == "4"
        assert result[0].content == "b2 (just committed)"
        assert result[0].parent_source_id == "3"

    def test_detects_chained_trailing_bots(self):
        """Multi-post bot turn: two consecutive bot messages, both unacknowledged."""
        stored = conversation(
            msg("1", "u1"),
            bot_msg("2", "b1-part-a"),
            bot_msg("3", "b1-part-b"),
        )
        partial = [_partial("u1", "1"), _partial("follow-up", None)]
        result = _detect_unacknowledged_bot_messages(stored, partial)
        assert len(result) == 2
        assert result[0].source_id == "2"
        assert result[0].parent_source_id == "1"  # first bot's parent is the user
        assert result[1].source_id == "3"
        assert result[1].parent_source_id == "2"  # chained: second bot's parent is the first bot

    def test_no_resync_when_trailing_is_user_message(self):
        """Stored trailing is a user message, not bot — not a resync scenario."""
        stored = conversation(
            msg("1", "u1"),
            bot_msg("2", "b1"),
            msg("3", "u2"),
        )
        # Client retries with [u1, follow-up], dropping u2 — divergence, not resync.
        partial = [_partial("u1", "1"), _partial("follow-up", None)]
        result = _detect_unacknowledged_bot_messages(stored, partial)
        assert result == []

    def test_no_resync_when_incoming_lacks_new_content(self):
        """Incoming is a strict sub-prefix of stored — no new content. Not a resync;
        falls through to divergence/delete classification."""
        stored = conversation(msg("1", "u1"), bot_msg("2", "b1"))
        partial = [_partial("u1", "1")]
        result = _detect_unacknowledged_bot_messages(stored, partial)
        assert result == []

    def test_tombstoned_messages_excluded_from_resync_check(self):
        """A tombstoned trailing bot doesn't trigger resync — it's no longer authoritative."""
        stored = conversation(
            msg("1", "u1"),
            bot_msg("2", "b1 (deleted)"),
            bot_msg("3", "b2"),
        )
        # Tombstone b1 (source_id 2)
        stored.messages[1].deleted = True
        partial = [_partial("u1", "1"), _partial("new", None)]
        result = _detect_unacknowledged_bot_messages(stored, partial)
        # b1 (deleted) excluded; b2 remains as the unacknowledged trailing bot
        assert len(result) == 1
        assert result[0].source_id == "3"
        assert result[0].parent_source_id == "1"


class TestPartialMessage:
    def test_unassigned_source_id_is_allowed(self):
        """Pre-assignment _PartialMessage may carry source_id=None."""
        p = _partial("hi", None)
        assert p.source_id is None
        assert p.content == "hi"
        assert p.author_id == "u-alice"
