"""Unit tests for the compactor's pure helpers (recency tail, lift plan, prefix match).

CAS-swap + ES integration tests live in the syncer/compactor integration suite once a fake SearchClient is
wired.
"""

from __future__ import annotations

from prokaryotes.context_v1.compaction import (
    _compute_lift_plan,
    _messages_match_prefix,
    _recency_tail_messages,
)
from prokaryotes.conversation_v1.models import TurnExecution, TurnItem
from tests.unit_tests._builders import BOT_ID, bot_msg, msg


def _file_call(call_id: str, path: str, status: str = "live") -> TurnItem:
    return TurnItem(
        type="function_call",
        call_id=call_id,
        name="FileTool",
        arguments=f'{{"path": "{path}"}}',
        prokaryotes_annotations={"file_tool.path": path, "file_tool.status": status},
    )


def _file_output(call_id: str, path: str, body: str, status: str = "live") -> TurnItem:
    return TurnItem(
        type="function_call_output",
        call_id=call_id,
        output=body,
        prokaryotes_annotations={"file_tool.path": path, "file_tool.status": status},
    )


def _turn(bot_message_source_id: str, *items: TurnItem) -> TurnExecution:
    return TurnExecution(
        conversation_uuid="c-1",
        bot_message_source_id=bot_message_source_id,
        items=list(items),
        completed=True,
    )


class TestRecencyTailMessages:
    def test_returns_last_n_non_deleted(self):
        messages = [
            msg("1", "u1"),
            bot_msg("2", "b1"),
            msg("3", "u2"),
            bot_msg("4", "b2"),
            msg("5", "u3"),
            bot_msg("6", "b3"),
        ]
        tail, offset = _recency_tail_messages(messages, BOT_ID, tail_count=4)
        # Last 4 non-deleted = [u2, b2, u3, b3]; tail leads with u2 (non-bot already)
        assert [m.content for m in tail] == ["u2", "b2", "u3", "b3"]
        assert offset == 2  # u1, b1 in pre-tail

    def test_advances_past_leading_bot(self):
        """Tail must lead with a non-bot message — provider constraint."""
        messages = [
            msg("1", "u1"),
            msg("2", "u2"),
            bot_msg("3", "b1"),
            bot_msg("4", "b2"),
            msg("5", "u3"),
            bot_msg("6", "b3"),
        ]
        tail, offset = _recency_tail_messages(messages, BOT_ID, tail_count=4)
        # Last 4 non-deleted starts at "b1" (bot); advance to next non-bot at "u3"
        assert [m.content for m in tail] == ["u3", "b3"]
        assert offset == 4  # u1, u2, b1, b2 in pre-tail

    def test_empty_when_only_bots_in_tail(self):
        messages = [msg("1", "u1"), bot_msg("2", "b1"), bot_msg("3", "b2")]
        tail, offset = _recency_tail_messages(messages, BOT_ID, tail_count=2)
        # Last 2 are both bots; advance past them returns empty
        assert tail == []
        assert offset == 0

    def test_handles_deleted_messages_in_count(self):
        messages = [
            msg("1", "u1"),
            msg("2", "deleted", deleted=True),
            bot_msg("3", "b1"),
            msg("4", "u2"),
            bot_msg("5", "b2"),
        ]
        tail, offset = _recency_tail_messages(messages, BOT_ID, tail_count=3)
        # Non-deleted: [u1, b1, u2, b2]; last 3 = [b1, u2, b2]; advance to u2
        assert [m.content for m in tail] == ["u2", "b2"]
        # tail_offset is non-deleted-count before tail: u1, b1 = 2
        assert offset == 2

    def test_empty_input(self):
        tail, offset = _recency_tail_messages([], BOT_ID, tail_count=5)
        assert tail == []
        assert offset == 0


class TestMessagesMatchPrefix:
    def test_exact_prefix_matches(self):
        snap = [msg("1", "a"), msg("2", "b")]
        current = [msg("1", "a"), msg("2", "b"), msg("3", "c")]
        assert _messages_match_prefix(current, snap)

    def test_shorter_current_does_not_match(self):
        snap = [msg("1", "a"), msg("2", "b")]
        current = [msg("1", "a")]
        assert not _messages_match_prefix(current, snap)

    def test_diverged_prefix_does_not_match(self):
        snap = [msg("1", "a"), msg("2", "b")]
        current = [msg("1", "a"), msg("2", "b-edited")]
        assert not _messages_match_prefix(current, snap)


class TestComputeLiftPlan:
    def test_no_lift_when_no_active_paths(self):
        """Empty recency-tail turns → empty active paths → empty lift plan."""
        plan = _compute_lift_plan(
            pre_tail_messages=[bot_msg("1", "b")],
            recency_tail_messages=[msg("2", "u")],
            pre_tail_turns={"1": _turn("1", _file_call("c1", "/a"), _file_output("c1", "/a", "<body>"))},
            recency_tail_turns={},
            parent_lifted=[],
            bot_author_id=BOT_ID,
        )
        assert plan.lifted_turn_items == []
        assert plan.lifted_anchor_source_id is None

    def test_lifts_pre_tail_pair_when_path_active_in_tail(self):
        plan = _compute_lift_plan(
            pre_tail_messages=[bot_msg("1", "b1")],
            recency_tail_messages=[msg("2", "u"), bot_msg("3", "b2")],
            pre_tail_turns={"1": _turn("1", _file_call("c1", "/a"), _file_output("c1", "/a", "<pre-tail body>"))},
            recency_tail_turns={
                "3": _turn("3", _file_call("c2", "/a"), _file_output("c2", "/a", "<tail body>")),
            },
            parent_lifted=[],
            bot_author_id=BOT_ID,
        )
        # Pre-tail pair for /a is superseded by fresh read in tail (path "/a" freshly read in tail)
        assert plan.lifted_turn_items == []

    def test_lifts_when_path_active_but_not_freshly_read(self):
        """Path appears in tail via an edit/annotation but not as a fresh read — pre-tail lifts."""
        plan = _compute_lift_plan(
            pre_tail_messages=[bot_msg("1", "b1")],
            recency_tail_messages=[msg("2", "u"), bot_msg("3", "b2")],
            pre_tail_turns={"1": _turn("1", _file_call("c1", "/a"), _file_output("c1", "/a", "<body>"))},
            recency_tail_turns={
                # Tail has a function_call annotating "/a" but no fresh read (no output)
                "3": _turn("3", _file_call("c2", "/a")),
            },
            parent_lifted=[],
            bot_author_id=BOT_ID,
        )
        # The pre-tail (function_call, function_call_output) pair is lifted
        assert len(plan.lifted_turn_items) == 2
        assert plan.lifted_turn_items[0].type == "function_call"
        assert plan.lifted_turn_items[1].type == "function_call_output"
        assert plan.lifted_anchor_source_id == "3"

    def test_anchor_is_first_bot_with_file_activity(self):
        """Anchor selects the *first* recency-tail bot whose TurnExecution has a file-tool annotation —
        placement is right before that turn's tool round."""
        plan = _compute_lift_plan(
            pre_tail_messages=[bot_msg("1", "b1")],
            recency_tail_messages=[
                msg("2", "u"),
                bot_msg("3", "b2 (no files)"),
                msg("4", "u2"),
                bot_msg("5", "b3 (touches a)"),
            ],
            pre_tail_turns={"1": _turn("1", _file_call("c1", "/a"), _file_output("c1", "/a", "<body>"))},
            recency_tail_turns={
                "3": _turn("3"),  # no items
                "5": _turn("5", _file_call("c2", "/a")),
            },
            parent_lifted=[],
            bot_author_id=BOT_ID,
        )
        assert plan.lifted_anchor_source_id == "5"

    def test_stale_pre_tail_pairs_excluded(self):
        plan = _compute_lift_plan(
            pre_tail_messages=[bot_msg("1", "b1")],
            recency_tail_messages=[msg("2", "u"), bot_msg("3", "b2")],
            pre_tail_turns={
                "1": _turn(
                    "1",
                    _file_call("c1", "/a", status="stale"),
                    _file_output("c1", "/a", "<tombstone>", status="stale"),
                ),
            },
            recency_tail_turns={"3": _turn("3", _file_call("c2", "/a"))},
            parent_lifted=[],
            bot_author_id=BOT_ID,
        )
        # Stale pair not lifted; with no eligible lift, anchor falls through to None (invariant: anchor=None
        # iff lifted_turn_items == [])
        assert plan.lifted_turn_items == []
        assert plan.lifted_anchor_source_id is None

    def test_parent_lifted_items_carry_forward_through_compaction(self):
        """Transitive roll-forward: items lifted by an earlier compaction re-lift here if their path is still
        active and not superseded."""
        parent_lifted = [
            _file_call("lifted-c1", "/keep"),
            _file_output("lifted-c1", "/keep", "<old body>"),
        ]
        plan = _compute_lift_plan(
            pre_tail_messages=[bot_msg("1", "b1")],
            recency_tail_messages=[msg("2", "u"), bot_msg("3", "b2")],
            pre_tail_turns={},
            recency_tail_turns={"3": _turn("3", _file_call("c2", "/keep"))},
            parent_lifted=parent_lifted,
            bot_author_id=BOT_ID,
        )
        assert len(plan.lifted_turn_items) == 2
        assert plan.lifted_turn_items[0].call_id == "lifted-c1"
        assert plan.lifted_anchor_source_id == "3"

    def test_multiple_windows_per_path_preserved(self):
        """Two distinct live windows over the same file are first-class — no per-path dedup."""
        plan = _compute_lift_plan(
            pre_tail_messages=[bot_msg("1", "b1"), bot_msg("2", "b2")],
            recency_tail_messages=[msg("3", "u"), bot_msg("4", "b3")],
            pre_tail_turns={
                "1": _turn(
                    "1",
                    _file_call("c1", "/a"),
                    _file_output("c1", "/a", "<window 1>"),
                ),
                "2": _turn(
                    "2",
                    _file_call("c2", "/a"),
                    _file_output("c2", "/a", "<window 2>"),
                ),
            },
            recency_tail_turns={"4": _turn("4", _file_call("c3", "/a"))},
            parent_lifted=[],
            bot_author_id=BOT_ID,
        )
        # Both windows lifted — 2 pairs
        assert len(plan.lifted_turn_items) == 4
        outputs = [item.output for item in plan.lifted_turn_items if item.type == "function_call_output"]
        assert outputs == ["<window 1>", "<window 2>"]
