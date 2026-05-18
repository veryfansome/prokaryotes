"""Tests for Case A lifted-state recompute on web branch creation.

When a web edit/regenerate produces a new sibling snapshot inside the parent's raw window, the child must:

- inherit `ancestor_summaries` and `raw_message_start_index` verbatim,
- filter parent's `lifted_turn_items` to paths still active in the *child's* raw window (shared prefix's
  TurnExecutions),
- pick `lifted_anchor_source_id` as the first non-tombstoned bot in the shared prefix with non-stale file-tool
  activity,
- preserve the invariant `anchor=None iff lifted_turn_items == []`.

Stub of `_active_paths_for_messages` previously caused this whole subsystem to silently drop lifted state on
every web branch. These tests guard the fix.
"""

from __future__ import annotations

import pytest

from prokaryotes.api_v1.models import IncomingMessage
from prokaryotes.conversation_v1.models import TurnExecution, TurnItem
from tests.unit_tests._builders import BOT_ID, bot_msg, conversation, msg
from tests.unit_tests._fakes import make_syncer


def _store_redis_snapshot(redis_client, conversation_obj):
    redis_client._store[f"conversation:{conversation_obj.conversation_uuid}"] = (
        conversation_obj.model_dump_json().encode("utf-8")
    )


def _file_call(call_id: str, path: str, *, status: str = "live") -> TurnItem:
    return TurnItem(
        type="function_call",
        call_id=call_id,
        name="FileTool",
        arguments=f'{{"path": "{path}"}}',
        prokaryotes_annotations={"file_tool.path": path, "file_tool.status": status},
    )


def _file_output(call_id: str, path: str, body: str, *, status: str = "live") -> TurnItem:
    return TurnItem(
        type="function_call_output",
        call_id=call_id,
        output=body,
        prokaryotes_annotations={"file_tool.path": path, "file_tool.status": status},
    )


class TestCaseABranchPreservesLifted:
    """Parent has lifted pair for `/a`; shared prefix's bot turn touches `/a`; new branch retains the lifted
    pair, anchored at the touching bot."""

    @pytest.mark.asyncio
    async def test_lifted_pair_carries_to_branch_when_path_active(self):
        syncer, redis, search = make_syncer()
        parent = conversation(
            msg("1.000000", "u1"),
            bot_msg("1.000001", "b1 (touches /a)"),
            msg("1.000002", "u2"),
            bot_msg("1.000003", "b2"),
            snapshot_uuid="s-parent",
            lifted_turn_items=[
                _file_call("lift-c1", "/a"),
                _file_output("lift-c1", "/a", "<lifted body>"),
            ],
            lifted_anchor_source_id="1.000001",
        )
        search.store_conversation_doc(parent)
        _store_redis_snapshot(redis, parent)
        # b1's TurnExecution shows ongoing /a activity.
        await search.put_turn_execution(
            TurnExecution(
                conversation_uuid=parent.conversation_uuid,
                bot_message_source_id="1.000001",
                items=[_file_call("c1", "/a"), _file_output("c1", "/a", "<b1 read>")],
                completed=True,
            )
        )

        # Web edit of u2 — regenerate from there; sends history up through u2. Reconcile classifies as
        # "delete" (omits b2), web policy branches.
        incoming = [
            IncomingMessage(role="user", content="u1", source_id="1.000000"),
            IncomingMessage(role="assistant", content="b1 (touches /a)", source_id="1.000001"),
            IncomingMessage(role="user", content="u2 EDITED", source_id="1.000002"),
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
        # Lifted pair retained on the branch.
        assert len(result.conversation.lifted_turn_items) == 2
        call, output = result.conversation.lifted_turn_items
        assert call.type == "function_call"
        assert output.type == "function_call_output"
        assert output.output == "<lifted body>"
        # Anchor is the bot in the shared prefix that touches /a.
        assert result.conversation.lifted_anchor_source_id == "1.000001"
        # Inherited compacted-prefix state.
        assert result.conversation.ancestor_summaries == parent.ancestor_summaries
        assert result.conversation.raw_message_start_index == parent.raw_message_start_index


class TestCaseABranchDropsWhenPathNotActive:
    """Parent has lifted pair for `/a`, but the shared prefix has no `/a` activity — drop the lifted pair,
    anchor stays None."""

    @pytest.mark.asyncio
    async def test_no_shared_prefix_activity_drops_lifted(self):
        syncer, redis, search = make_syncer()
        parent = conversation(
            msg("1.000000", "u1"),
            bot_msg("1.000001", "b1 (touches /b)"),
            msg("1.000002", "u2"),
            bot_msg("1.000003", "b2 (touches /a)"),
            snapshot_uuid="s-parent",
            lifted_turn_items=[
                _file_call("lift-c1", "/a"),
                _file_output("lift-c1", "/a", "<lifted body>"),
            ],
            lifted_anchor_source_id="1.000003",  # /a anchor is on b2
        )
        search.store_conversation_doc(parent)
        _store_redis_snapshot(redis, parent)
        # b1 touches /b; b2 touches /a. The shared prefix on regenerate-from-u2 is [u1, b1, u2] — only b1's
        # turn matters.
        await search.put_turn_execution(
            TurnExecution(
                conversation_uuid=parent.conversation_uuid,
                bot_message_source_id="1.000001",
                items=[_file_call("c1", "/b"), _file_output("c1", "/b", "<b1 read>")],
                completed=True,
            )
        )
        await search.put_turn_execution(
            TurnExecution(
                conversation_uuid=parent.conversation_uuid,
                bot_message_source_id="1.000003",
                items=[_file_call("c2", "/a"), _file_output("c2", "/a", "<b2 read>")],
                completed=True,
            )
        )

        # Regenerate from u2 — shared prefix = [u1, b1, u2]; b2 is dropped.
        incoming = [
            IncomingMessage(role="user", content="u1", source_id="1.000000"),
            IncomingMessage(role="assistant", content="b1 (touches /b)", source_id="1.000001"),
            IncomingMessage(role="user", content="u2", source_id="1.000002"),
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
        # /a isn't active in the shared prefix — lifted pair dropped.
        assert result.conversation.lifted_turn_items == []
        # Anchor invariant: anchor=None iff lifted==[].
        assert result.conversation.lifted_anchor_source_id is None


class TestCaseABranchMultipleWindowsPerPath:
    """Parent has two distinct lifted pairs for `/a` (different view ranges); branch retains both."""

    @pytest.mark.asyncio
    async def test_both_windows_carry_forward(self):
        syncer, redis, search = make_syncer()
        parent = conversation(
            msg("1.000000", "u1"),
            bot_msg("1.000001", "b1 (touches /a)"),
            msg("1.000002", "u2"),
            snapshot_uuid="s-parent",
            lifted_turn_items=[
                _file_call("lift-c1", "/a"),
                _file_output("lift-c1", "/a", "<window 1>"),
                _file_call("lift-c2", "/a"),
                _file_output("lift-c2", "/a", "<window 2>"),
            ],
            lifted_anchor_source_id="1.000001",
        )
        search.store_conversation_doc(parent)
        _store_redis_snapshot(redis, parent)
        await search.put_turn_execution(
            TurnExecution(
                conversation_uuid=parent.conversation_uuid,
                bot_message_source_id="1.000001",
                items=[_file_call("c1", "/a"), _file_output("c1", "/a", "<b1 read>")],
                completed=True,
            )
        )

        # Edit u2 — shared prefix = [u1, b1] (u2 content changes).
        incoming = [
            IncomingMessage(role="user", content="u1", source_id="1.000000"),
            IncomingMessage(role="assistant", content="b1 (touches /a)", source_id="1.000001"),
            IncomingMessage(role="user", content="u2 edited", source_id="1.000002"),
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
        # Both lifted pairs preserved — multiple windows over the same path is first-class.
        outputs = [item.output for item in result.conversation.lifted_turn_items if item.type == "function_call_output"]
        assert outputs == ["<window 1>", "<window 2>"]
        assert result.conversation.lifted_anchor_source_id == "1.000001"


class TestCaseAAnchorPicksFirstNonTombstonedBot:
    """When the shared prefix has multiple bots with file activity, the anchor is the FIRST non-tombstoned one
    (preserving today's placement rule)."""

    @pytest.mark.asyncio
    async def test_anchor_is_first_qualifying_bot(self):
        syncer, redis, search = make_syncer()
        parent = conversation(
            msg("1.000000", "u1"),
            bot_msg("1.000001", "b1 (no files)"),
            msg("1.000002", "u2"),
            bot_msg("1.000003", "b2 (touches /a)"),
            msg("1.000004", "u3"),
            bot_msg("1.000005", "b3 (touches /a)"),
            snapshot_uuid="s-parent",
            lifted_turn_items=[
                _file_call("lift-c1", "/a"),
                _file_output("lift-c1", "/a", "<lifted>"),
            ],
            lifted_anchor_source_id="1.000003",
        )
        search.store_conversation_doc(parent)
        _store_redis_snapshot(redis, parent)
        # b1 has no items; b2 and b3 both touch /a.
        await search.put_turn_execution(
            TurnExecution(
                conversation_uuid=parent.conversation_uuid,
                bot_message_source_id="1.000001",
                items=[],
                completed=True,
            )
        )
        await search.put_turn_execution(
            TurnExecution(
                conversation_uuid=parent.conversation_uuid,
                bot_message_source_id="1.000003",
                items=[_file_call("c2", "/a"), _file_output("c2", "/a", "<b2 read>")],
                completed=True,
            )
        )
        await search.put_turn_execution(
            TurnExecution(
                conversation_uuid=parent.conversation_uuid,
                bot_message_source_id="1.000005",
                items=[_file_call("c3", "/a"), _file_output("c3", "/a", "<b3 read>")],
                completed=True,
            )
        )

        # Edit u3 — shared prefix = [u1, b1, u2, b2]. Both b1 and b2 are in the shared prefix; b1 has no file
        # activity, b2 has /a activity.
        incoming = [
            IncomingMessage(role="user", content="u1", source_id="1.000000"),
            IncomingMessage(role="assistant", content="b1 (no files)", source_id="1.000001"),
            IncomingMessage(role="user", content="u2", source_id="1.000002"),
            IncomingMessage(role="assistant", content="b2 (touches /a)", source_id="1.000003"),
            IncomingMessage(role="user", content="u3 edited", source_id="1.000004"),
        ]
        result = await syncer.sync_conversation(
            conversation_uuid=parent.conversation_uuid,
            snapshot_uuid="s-parent",
            bot_author_id=BOT_ID,
            incoming=incoming,
            session_user_id="u-alice",
            session_display_name="Alice",
        )

        # b2 is the first qualifying bot in the shared prefix.
        assert result.conversation.lifted_anchor_source_id == "1.000003"


class TestCaseAInvariantHoldsWithNoSharedBots:
    """No shared bots (divergence right at the start): no shared turns to load, no active paths, lifted state
    is dropped per the invariant."""

    @pytest.mark.asyncio
    async def test_empty_shared_prefix_drops_lifted(self):
        syncer, redis, search = make_syncer()
        parent = conversation(
            msg("1.000000", "u1"),
            bot_msg("1.000001", "b1"),
            snapshot_uuid="s-parent",
            lifted_turn_items=[
                _file_call("lift-c1", "/a"),
                _file_output("lift-c1", "/a", "<lifted>"),
            ],
            lifted_anchor_source_id="1.000001",
        )
        search.store_conversation_doc(parent)
        _store_redis_snapshot(redis, parent)

        # Client edits u1 — no shared prefix at all.
        incoming = [
            IncomingMessage(role="user", content="u1 edited", source_id="1.000000"),
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
        assert result.conversation.lifted_turn_items == []
        assert result.conversation.lifted_anchor_source_id is None
