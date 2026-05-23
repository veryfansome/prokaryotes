"""Tests for surface-specific apply policy.

Web (default `ConversationSyncer._apply_result`): edit, delete, divergence all branch via `_apply_divergence`.
The parent snapshot stays intact.

Slack (`SlackApplyPolicy._apply_result`): edit / delete / divergence mutate the stored snapshot in
place. Delete also triggers tombstone re-keying of the bot's `TurnExecution` ownership (the lifted-anchor
re-key the previous design carried is gone — working-file state is no longer anchor-positioned by bot
source_id).
"""

from __future__ import annotations

import pytest

from prokaryotes.api_v1.models import IncomingMessage
from prokaryotes.context_v1.conversation_sync import (
    ConversationSyncer,
    SlackApplyPolicy,
    _next_non_tombstoned_bot_in_run,
)
from prokaryotes.conversation_v1.models import TurnExecution, TurnItem
from tests.unit_tests._builders import BOT_ID, bot_msg, conversation, msg
from tests.unit_tests._fakes import FakeRedis, FakeSearchClient, make_syncer


def _store_redis_snapshot(redis_client, conversation_obj):
    redis_client._store[f"conversation:{conversation_obj.conversation_uuid}"] = (
        conversation_obj.model_dump_json().encode("utf-8")
    )


class _SlackSyncerForTest(SlackApplyPolicy, ConversationSyncer):
    """Concrete Slack syncer wired to fakes for tests."""

    def __init__(self, *, redis_client, search_client):
        self._redis_client = redis_client
        self._search_client = search_client
        self._conversation_cache_ex = 60 * 60 * 24 * 7

    @property
    def conversation_cache_ex(self) -> int:
        return self._conversation_cache_ex

    @property
    def redis_client(self):  # type: ignore[override]
        return self._redis_client

    @property
    def search_client(self):  # type: ignore[override]
        return self._search_client


def _make_slack_syncer() -> tuple[_SlackSyncerForTest, FakeRedis, FakeSearchClient]:
    redis = FakeRedis()
    search = FakeSearchClient()
    syncer = _SlackSyncerForTest(redis_client=redis, search_client=search)
    return syncer, redis, search


class TestWebBranchesOnEdit:
    """Web default: an edit to a stored message produces a new sibling snapshot; the parent snapshot stays
    intact with its original content."""

    @pytest.mark.asyncio
    async def test_edit_creates_branch_snapshot(self):
        syncer, redis, search = make_syncer()
        parent = conversation(
            msg("1717000001.000000", "u1"),
            bot_msg("1717000002.000000", "b1"),
            msg("1717000003.000000", "u2"),
            snapshot_uuid="s-parent",
        )
        search.store_conversation_doc(parent)
        _store_redis_snapshot(redis, parent)

        # Client edits U2 (same source_id, different content).
        incoming = [
            IncomingMessage(role="user", content="u1", source_id="1717000001.000000"),
            IncomingMessage(role="assistant", content="b1", source_id="1717000002.000000"),
            IncomingMessage(role="user", content="u2 edited", source_id="1717000003.000000"),
        ]
        result = await syncer.sync_conversation(
            conversation_uuid=parent.conversation_uuid,
            snapshot_uuid="s-parent",
            bot_author_id=BOT_ID,
            incoming=incoming,
            session_user_id="u-alice",
            session_display_name="Alice",
        )

        # New branch.
        assert result.is_new_branch is True
        assert result.conversation.snapshot_uuid != "s-parent"
        assert result.conversation.parent_snapshot_uuid == "s-parent"
        # New branch carries the edited content.
        contents = [m.content for m in result.conversation.messages]
        assert "u2 edited" in contents
        assert "u2" not in contents  # only the edited version survived

        # Parent snapshot in ES is unchanged — original "u2" still there.
        parent_doc = search.conversations["s-parent"]
        import json

        parent_msgs = json.loads(parent_doc["messages_json"])["messages"]
        parent_contents = [m["content"] for m in parent_msgs]
        assert "u2" in parent_contents
        assert "u2 edited" not in parent_contents


class TestWebBranchesOnDelete:
    """Web default: regenerate (omitting the trailing assistant message) produces a new sibling snapshot. The
    parent retains the deleted bot in non-tombstoned state."""

    @pytest.mark.asyncio
    async def test_regenerate_creates_branch_with_bot_omitted(self):
        syncer, redis, search = make_syncer()
        parent = conversation(
            msg("1717000001.000000", "u1"),
            bot_msg("1717000002.000000", "b1"),
            msg("1717000003.000000", "u2"),
            bot_msg("1717000004.000000", "b2"),
            snapshot_uuid="s-parent",
        )
        search.store_conversation_doc(parent)
        _store_redis_snapshot(redis, parent)

        # Regenerate from U2: client sends history up to U2 only.
        incoming = [
            IncomingMessage(role="user", content="u1", source_id="1717000001.000000"),
            IncomingMessage(role="assistant", content="b1", source_id="1717000002.000000"),
            IncomingMessage(role="user", content="u2", source_id="1717000003.000000"),
        ]
        result = await syncer.sync_conversation(
            conversation_uuid=parent.conversation_uuid,
            snapshot_uuid="s-parent",
            bot_author_id=BOT_ID,
            incoming=incoming,
            session_user_id="u-alice",
            session_display_name="Alice",
        )

        assert result.is_new_branch is True
        assert result.conversation.snapshot_uuid != "s-parent"
        assert result.conversation.parent_snapshot_uuid == "s-parent"
        # New branch has no B2.
        branch_source_ids = [m.source_id for m in result.conversation.messages]
        assert "1717000004.000000" not in branch_source_ids

        # Parent's B2 is unchanged — not tombstoned in place.
        import json

        parent_doc = search.conversations["s-parent"]
        parent_msgs = json.loads(parent_doc["messages_json"])["messages"]
        b2 = next(m for m in parent_msgs if m["source_id"] == "1717000004.000000")
        assert b2["deleted"] is False


class TestSlackInPlaceEdit:
    """Slack: `message_changed` mutates content in place; no new snapshot."""

    @pytest.mark.asyncio
    async def test_edit_mutates_stored_in_place(self):
        syncer, redis, search = _make_slack_syncer()
        stored = conversation(
            msg("1.000000", "original", author_id="u-alice"),
            bot_msg("1.000001", "reply"),
            snapshot_uuid="s-slack",
        )
        search.store_conversation_doc(stored)
        _store_redis_snapshot(redis, stored)

        incoming = [
            IncomingMessage(role="user", content="edited", source_id="1.000000"),
            IncomingMessage(role="assistant", content="reply", source_id="1.000001"),
        ]
        result = await syncer.sync_conversation(
            conversation_uuid=stored.conversation_uuid,
            snapshot_uuid="s-slack",
            bot_author_id=BOT_ID,
            incoming=incoming,
            session_user_id="u-alice",
            session_display_name="Alice",
        )

        # Same snapshot.
        assert result.is_new_branch is False
        assert result.conversation.snapshot_uuid == "s-slack"
        # Content mutated, edited flag set.
        msg_edited = result.conversation.message_by_source_id("1.000000")
        assert msg_edited.content == "edited"
        assert msg_edited.edited is True


class TestSlackInPlaceDelete:
    """Slack: `message_deleted` sets the tombstone in place AND re-keys the bot's `TurnExecution` to the next
    non-tombstoned bot in the same run."""

    @pytest.mark.asyncio
    async def test_delete_tombstone_in_place(self):
        """Trailing bot delete: tombstone in place, no new snapshot."""
        syncer, redis, search = _make_slack_syncer()
        stored = conversation(
            msg("1.000000", "u1"),
            bot_msg("1.000001", "b1"),
            msg("1.000002", "u2"),
            bot_msg("1.000003", "b2"),
            snapshot_uuid="s-slack",
        )
        search.store_conversation_doc(stored)
        _store_redis_snapshot(redis, stored)

        # Client omits b2 → reconcile classifies as delete (trailing).
        incoming = [
            IncomingMessage(role="user", content="u1", source_id="1.000000"),
            IncomingMessage(role="assistant", content="b1", source_id="1.000001"),
            IncomingMessage(role="user", content="u2", source_id="1.000002"),
        ]
        result = await syncer.sync_conversation(
            conversation_uuid=stored.conversation_uuid,
            snapshot_uuid="s-slack",
            bot_author_id=BOT_ID,
            incoming=incoming,
            session_user_id="u-alice",
            session_display_name="Alice",
        )
        assert result.is_new_branch is False
        b2 = result.conversation.message_by_source_id("1.000003")
        assert b2.deleted is True


class TestSlackDivergenceOverwrites:
    @pytest.mark.asyncio
    async def test_divergence_overwrites_in_place(self):
        """Slack reconciling against a stored snapshot that drifted — the recovery action is overwrite in
        place, not branch."""
        syncer, redis, search = _make_slack_syncer()
        stored = conversation(
            msg("1.000000", "u1-old"),
            bot_msg("1.000001", "b1-old"),
            msg("1.000002", "u2-old"),
            snapshot_uuid="s-slack",
        )
        search.store_conversation_doc(stored)
        _store_redis_snapshot(redis, stored)

        # Incoming has a divergence (m1 content differs and entries are partly missing).
        incoming = [
            IncomingMessage(role="user", content="u1-canonical", source_id="1.000000"),
            IncomingMessage(role="assistant", content="b1-canonical", source_id="1.000001"),
        ]
        result = await syncer.sync_conversation(
            conversation_uuid=stored.conversation_uuid,
            snapshot_uuid="s-slack",
            bot_author_id=BOT_ID,
            incoming=incoming,
            session_user_id="u-alice",
            session_display_name="Alice",
        )
        # Same snapshot, content reconciled in place. The mixed-kind divergence applies as
        # edit(u1) + edit(b1) + delete(u2) — u2 is tombstoned rather than dropped from the list.
        assert result.is_new_branch is False
        assert result.conversation.snapshot_uuid == "s-slack"
        assert [m.content for m in result.conversation.messages if not m.deleted] == [
            "u1-canonical",
            "b1-canonical",
        ]
        u2 = result.conversation.message_by_source_id("1.000002")
        assert u2 is not None and u2.deleted is True


class TestTombstoneRekey:
    """`_rekey_for_tombstone` and `_next_non_tombstoned_bot_in_run`: the bot run walking + ES re-key behavior
    the Slack delete path uses."""

    @pytest.mark.asyncio
    async def test_rekey_single_bot_run_orphans_turn_execution(self):
        """Bot run of one: tombstoning it leaves no replacement, so the `TurnExecution` is deleted."""
        syncer, _redis, search = _make_slack_syncer()
        stored = conversation(
            msg("1.000000", "u1"),
            bot_msg("1.000001", "b1"),
            snapshot_uuid="s-slack",
        )
        # Pre-seed the TurnExecution for b1.
        await search.put_turn_execution(
            TurnExecution(
                conversation_uuid=stored.conversation_uuid,
                bot_message_source_id="1.000001",
                items=[TurnItem(type="function_call", call_id="c1", name="ft")],
                completed=True,
            )
        )

        # Mark b1 deleted, then call the re-key helper directly.
        stored.message_by_source_id("1.000001").deleted = True
        await syncer._rekey_for_tombstone(stored, "1.000001")

        # No replacement bot → TurnExecution deleted from ES.
        assert search.delete_turn_execution_calls == [("c-1", "1.000001")]
        assert search.rekey_turn_execution_calls == []

    @pytest.mark.asyncio
    async def test_rekey_chain_promotes_next_bot(self):
        """Multi-post bot run (2 bots between user messages): tombstoning the first bot moves its
        `TurnExecution` to the second bot's source_id."""
        syncer, _redis, search = _make_slack_syncer()
        stored = conversation(
            msg("1.000000", "u1"),
            bot_msg("1.000001", "b1-part-a"),
            bot_msg("1.000002", "b1-part-b"),
            snapshot_uuid="s-slack",
        )
        await search.put_turn_execution(
            TurnExecution(
                conversation_uuid=stored.conversation_uuid,
                bot_message_source_id="1.000001",
                items=[TurnItem(type="function_call", call_id="c1", name="ft")],
                completed=True,
            )
        )

        # Tombstone b1-part-a; rekey runs.
        stored.message_by_source_id("1.000001").deleted = True
        await syncer._rekey_for_tombstone(stored, "1.000001")

        # ES doc moved to b1-part-b's source_id.
        assert search.rekey_turn_execution_calls == [("c-1", "1.000001", "1.000002")]
        assert search.delete_turn_execution_calls == []
        assert ("c-1", "1.000001") not in search.turn_executions
        assert ("c-1", "1.000002") in search.turn_executions

    @pytest.mark.asyncio
    async def test_rekey_full_run_tombstone_orphans(self):
        """Bot run of two with both already tombstoned by the time the helper runs: no replacement →
        `TurnExecution` deleted."""
        syncer, _redis, search = _make_slack_syncer()
        stored = conversation(
            msg("1.000000", "u1"),
            bot_msg("1.000001", "b1-part-a"),
            bot_msg("1.000002", "b1-part-b"),
            snapshot_uuid="s-slack",
        )
        await search.put_turn_execution(
            TurnExecution(
                conversation_uuid=stored.conversation_uuid,
                bot_message_source_id="1.000001",
                items=[],
                completed=True,
            )
        )
        # Both tombstoned.
        stored.message_by_source_id("1.000001").deleted = True
        stored.message_by_source_id("1.000002").deleted = True

        await syncer._rekey_for_tombstone(stored, "1.000001")

        assert search.delete_turn_execution_calls == [("c-1", "1.000001")]
        assert search.rekey_turn_execution_calls == []

    def test_user_tombstone_no_rekey(self):
        """Tombstoning a user message — `_next_non_tombstoned_bot_in_run` returns None because the deleted
        message isn't a bot (no run to walk)."""
        stored = conversation(
            msg("1.000000", "u1"),
            bot_msg("1.000001", "b1"),
            msg("1.000002", "u2"),
            snapshot_uuid="s-slack",
        )
        # Walk from a user source_id — there's no bot run *forward* (the next entry is u2, breaking the run),
        # and no bot *backward* before the user either. Returns None.
        stored.message_by_source_id("1.000000").deleted = True
        result = _next_non_tombstoned_bot_in_run(stored, "1.000000")
        assert result is None

    def test_rekey_skips_already_deleted_bots_in_run(self):
        """Bot run of three with middle already tombstoned: tombstoning the first re-keys past the tombstoned
        middle to the third bot (membership includes the tombstoned middle but selection skips it)."""
        stored = conversation(
            msg("1.000000", "u1"),
            bot_msg("1.000001", "b1-part-a"),
            bot_msg("1.000002", "b1-part-b"),
            bot_msg("1.000003", "b1-part-c"),
            snapshot_uuid="s-slack",
        )
        # Middle already tombstoned.
        stored.message_by_source_id("1.000002").deleted = True
        # Now tombstone the first; walker should land on the third (skipping the second).
        stored.message_by_source_id("1.000001").deleted = True
        result = _next_non_tombstoned_bot_in_run(stored, "1.000001")
        assert result == "1.000003"

    @pytest.mark.asyncio
    async def test_rekey_bot_without_turn_execution_no_op(self):
        """Tombstoning a non-owner bot (a continuation post that never had its own `TurnExecution` — only the
        first post in a run owns one) doesn't touch ES."""
        syncer, _redis, search = _make_slack_syncer()
        stored = conversation(
            msg("1.000000", "u1"),
            bot_msg("1.000001", "b1-part-a"),
            bot_msg("1.000002", "b1-part-b"),
            snapshot_uuid="s-slack",
        )
        # Only b1-part-a owns a TurnExecution; b1-part-b doesn't.
        await search.put_turn_execution(
            TurnExecution(
                conversation_uuid=stored.conversation_uuid,
                bot_message_source_id="1.000001",
                items=[],
                completed=True,
            )
        )
        search.delete_turn_execution_calls.clear()
        search.rekey_turn_execution_calls.clear()

        # Tombstone the non-owner.
        stored.message_by_source_id("1.000002").deleted = True
        await syncer._rekey_for_tombstone(stored, "1.000002")

        # b1-part-a's TurnExecution is untouched.
        assert search.delete_turn_execution_calls == []
        assert search.rekey_turn_execution_calls == []
        assert ("c-1", "1.000001") in search.turn_executions
