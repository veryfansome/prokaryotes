"""Unit tests for `ConversationSyncer._split_compacted_prefix`.

The split runs before resync detection and reconcile when stored has `raw_message_start_index > 0`. It
reconstructs the compacted-away prefix from the parent chain and matches it against the first N non-deleted
entries of incoming. Match → strip the prefix; mismatch → caller routes to Case B.

Comparison is `(source_id, author_id, content)` — same-content / different source_id routes to Case B
(preserves source-ID-as-identity invariant).
"""

from __future__ import annotations

import pytest

from prokaryotes.context_v1.conversation_sync import _PartialMessage
from tests.unit_tests._builders import BOT_ID, bot_msg, conversation, msg
from tests.unit_tests._fakes import make_syncer


def _partial(
    content: str,
    source_id: str | None,
    *,
    author_id: str = "u-alice",
    client_index: int = 0,
) -> _PartialMessage:
    return _PartialMessage(
        author_id=author_id,
        content=content,
        client_index=client_index,
        source_id=source_id,
    )


def _setup_post_compaction_state(search_client, *, parent_messages):
    """Build a compacted-parent + post-compaction-child pair, store both in the fake search client. Returns the
    child Conversation (the syncer's `stored`).
    """
    parent = conversation(
        *parent_messages,
        snapshot_uuid="s-parent",
        ancestor_summaries=[],
    )
    search_client.store_conversation_doc(parent, is_compacted=True, summary="<summary>")

    # Child: post-compaction tail. raw_message_start_index counts non-deleted parent messages that were
    # summarized away.
    child_messages = parent_messages[-2:]
    raw_start = len(parent_messages) - len(child_messages)
    child = conversation(
        *child_messages,
        snapshot_uuid="s-child",
        parent_snapshot_uuid="s-parent",
        ancestor_summaries=["<summary>"],
        raw_message_start_index=raw_start,
    )
    search_client.store_conversation_doc(child)
    return child


class TestMatchingPrefixStripsCleanly:
    """Happy path — incoming has prefix + raw window + new entry; split returns the raw-window suffix
    only."""

    @pytest.mark.asyncio
    async def test_returns_raw_suffix(self):
        syncer, _redis, search = make_syncer()
        stored = _setup_post_compaction_state(
            search,
            parent_messages=[
                msg("1", "u1"),
                bot_msg("2", "b1"),
                msg("3", "u2"),
                bot_msg("4", "b2"),
            ],
        )
        # Parent had 4 messages; child has the last 2 ([m3=u2, m4=b2]). raw_message_start_index=2. Client
        # echoes all 4 + a new user message.
        partial = [
            _partial("u1", "1", author_id="u-alice", client_index=0),
            _partial("b1", "2", author_id=BOT_ID, client_index=1),
            _partial("u2", "3", author_id="u-alice", client_index=2),
            _partial("b2", "4", author_id=BOT_ID, client_index=3),
            _partial("new", None, author_id="u-alice", client_index=4),
        ]
        match, suffix = await syncer._split_compacted_prefix(stored, partial)
        assert match is True
        assert [(p.source_id, p.content) for p in suffix] == [
            ("3", "u2"),
            ("4", "b2"),
            (None, "new"),
        ]
        # Original partial untouched on success path (returns a slice).
        assert len(partial) == 5

    @pytest.mark.asyncio
    async def test_zero_raw_start_short_circuits(self):
        """No prefix to validate — return (True, partial) unchanged."""
        syncer, _redis, search = make_syncer()
        stored = conversation(msg("1", "u1"), raw_message_start_index=0)
        search.store_conversation_doc(stored)
        partial = [_partial("u1", "1"), _partial("new", None, client_index=1)]
        match, suffix = await syncer._split_compacted_prefix(stored, partial)
        assert match is True
        assert suffix is partial


class TestContentMismatchRoutesToCaseB:
    @pytest.mark.asyncio
    async def test_differing_content_returns_false(self):
        syncer, _redis, search = make_syncer()
        stored = _setup_post_compaction_state(
            search,
            parent_messages=[
                msg("1", "u1"),
                bot_msg("2", "b1"),
                msg("3", "u2"),
                bot_msg("4", "b2"),
            ],
        )
        partial = [
            _partial("u1 EDITED", "1", client_index=0),
            _partial("b1", "2", author_id=BOT_ID, client_index=1),
            _partial("u2", "3", client_index=2),
            _partial("b2", "4", author_id=BOT_ID, client_index=3),
            _partial("new", None, client_index=4),
        ]
        match, returned = await syncer._split_compacted_prefix(stored, partial)
        assert match is False
        assert returned is partial


class TestSourceIdMismatchRoutesToCaseB:
    """Content matches but source_id differs — must route to Case B to preserve the source-ID-as-identity
    invariant."""

    @pytest.mark.asyncio
    async def test_same_content_different_source_id_fails(self):
        syncer, _redis, search = make_syncer()
        stored = _setup_post_compaction_state(
            search,
            parent_messages=[
                msg("1", "u1"),
                bot_msg("2", "b1"),
                msg("3", "u2"),
                bot_msg("4", "b2"),
            ],
        )
        partial = [
            # Client has lost track of m1's source_id — invented a new one.
            _partial("u1", "ALT-1", client_index=0),
            _partial("b1", "2", author_id=BOT_ID, client_index=1),
            _partial("u2", "3", client_index=2),
            _partial("b2", "4", author_id=BOT_ID, client_index=3),
        ]
        match, _ = await syncer._split_compacted_prefix(stored, partial)
        assert match is False


class TestLengthMismatchRoutesToCaseB:
    @pytest.mark.asyncio
    async def test_incoming_shorter_than_raw_start_fails(self):
        """Client sent fewer entries than the compacted prefix needs."""
        syncer, _redis, search = make_syncer()
        stored = _setup_post_compaction_state(
            search,
            parent_messages=[
                msg("1", "u1"),
                bot_msg("2", "b1"),
                msg("3", "u2"),
                bot_msg("4", "b2"),
            ],
        )
        partial = [_partial("u1", "1", client_index=0)]
        match, _ = await syncer._split_compacted_prefix(stored, partial)
        assert match is False

    @pytest.mark.asyncio
    async def test_missing_parent_doc_fails_safely(self):
        """Parent chain isn't reachable — split treats as mismatch."""
        syncer, _redis, _search = make_syncer()
        stored = conversation(
            msg("3", "u2"),
            bot_msg("4", "b2"),
            snapshot_uuid="s-orphan",
            parent_snapshot_uuid="s-missing",
            raw_message_start_index=2,
        )
        partial = [
            _partial("u1", "1", client_index=0),
            _partial("b1", "2", author_id=BOT_ID, client_index=1),
            _partial("u2", "3", client_index=2),
            _partial("b2", "4", author_id=BOT_ID, client_index=3),
        ]
        match, _ = await syncer._split_compacted_prefix(stored, partial)
        assert match is False


class TestWalksMultiCompactionChain:
    """Three-deep chain: root → child1 (compacted away m1..m4) → child2 (compacted away m1..m6). Split for
    child2 should reconstruct the first 6 messages from the global history walking through child1."""

    @pytest.mark.asyncio
    async def test_three_deep_chain(self):
        syncer, _redis, search = make_syncer()

        root_messages = [
            msg("1", "u1"),
            bot_msg("2", "b1"),
            msg("3", "u2"),
            bot_msg("4", "b2"),
            msg("5", "u3"),
            bot_msg("6", "b3"),
            msg("7", "u4"),
            bot_msg("8", "b4"),
        ]
        root = conversation(
            *root_messages,
            snapshot_uuid="s-root",
            ancestor_summaries=[],
            raw_message_start_index=0,
        )
        search.store_conversation_doc(root, is_compacted=True, summary="<sum-1>")

        # First compaction summarized m1..m4 → child1 has tail [m5..m8].
        child1 = conversation(
            *root_messages[4:],
            snapshot_uuid="s-child1",
            parent_snapshot_uuid="s-root",
            ancestor_summaries=["<sum-1>"],
            raw_message_start_index=4,
        )
        search.store_conversation_doc(child1, is_compacted=True, summary="<sum-2>")

        # Second compaction summarized m5,m6 → child2 has tail [m7,m8].
        child2 = conversation(
            *root_messages[6:],
            snapshot_uuid="s-child2",
            parent_snapshot_uuid="s-child1",
            ancestor_summaries=["<sum-1>", "<sum-2>"],
            raw_message_start_index=6,
        )
        search.store_conversation_doc(child2)

        # Client echoes the full 8-message history + a new entry.
        partial = [
            _partial("u1", "1", client_index=0),
            _partial("b1", "2", author_id=BOT_ID, client_index=1),
            _partial("u2", "3", client_index=2),
            _partial("b2", "4", author_id=BOT_ID, client_index=3),
            _partial("u3", "5", client_index=4),
            _partial("b3", "6", author_id=BOT_ID, client_index=5),
            _partial("u4", "7", client_index=6),
            _partial("b4", "8", author_id=BOT_ID, client_index=7),
            _partial("u5", None, client_index=8),
        ]
        match, suffix = await syncer._split_compacted_prefix(child2, partial)
        assert match is True
        # First 6 stripped; the suffix is [u4, b4, u5].
        assert [(p.source_id, p.content) for p in suffix] == [
            ("7", "u4"),
            ("8", "b4"),
            (None, "u5"),
        ]
