from __future__ import annotations

from prokaryotes.conversation_v1.project import project_for_llm
from tests.unit_tests._builders import (
    bot_msg,
    conversation,
    function_call,
    function_call_output,
    msg,
    turn,
)


class TestRoleDerivation:
    def test_bot_author_maps_to_assistant(self):
        c = conversation(msg("1", "hi"), bot_msg("2", "hello"))
        items = project_for_llm(c)
        assert [(p.role, p.content) for p in items] == [
            ("user", "hi"),
            ("assistant", "hello"),
        ]


class TestMultiAuthorPrefix:
    def test_no_prefix_when_single_human_author(self):
        c = conversation(
            msg("1", "first", author_id="u-alice", display_name="Alice"),
            msg("2", "second", author_id="u-alice", display_name="Alice"),
            bot_msg("3", "bot"),
        )
        items = project_for_llm(c)
        user_contents = [p.content for p in items if p.role == "user"]
        # Same author → merged + no prefix
        assert user_contents == ["first\n\nsecond"]

    def test_prefix_when_two_humans(self):
        c = conversation(
            msg("1", "hi from alice", author_id="u-alice", display_name="Alice"),
            msg("2", "hi from bob", author_id="u-bob", display_name="Bob"),
            bot_msg("3", "bot"),
        )
        items = project_for_llm(c)
        # Both user-role messages get prefixed; consecutive user merges with \n\n
        assert items[0].role == "user"
        assert items[0].content == "<Alice> hi from alice\n\n<Bob> hi from bob"
        assert items[1].role == "assistant"

    def test_no_prefix_when_display_name_missing(self):
        c = conversation(
            msg("1", "hi from alice", author_id="u-alice", display_name="Alice"),
            msg("2", "hi from bob", author_id="u-bob", display_name=None),
        )
        items = project_for_llm(c)
        # u-bob has no display name; only u-alice gets prefixed
        assert items[0].content == "<Alice> hi from alice\n\nhi from bob"


class TestSameRoleMerge:
    def test_two_consecutive_assistant_messages_merge(self):
        """Multi-post bot turn (Slack split): both posts stored as separate
        ConversationMessages with bot author. Projection merges them."""
        c = conversation(
            msg("1", "ask"),
            bot_msg("2", "part-a"),
            bot_msg("3", "part-b"),
        )
        items = project_for_llm(c)
        assistant_items = [p for p in items if p.role == "assistant"]
        assert len(assistant_items) == 1
        assert assistant_items[0].content == "part-a\n\npart-b"

    def test_function_call_breaks_merge(self):
        """Function-call items break the same-role merge run."""
        c = conversation(msg("1", "ask"), bot_msg("2", "part-a"), bot_msg("3", "part-b"))
        historical_turns = {
            "3": turn(
                "3",
                function_call("call-1", "ThinkTool"),
                function_call_output("call-1", "<thought>"),
            ),
        }
        items = project_for_llm(c, historical_turns)
        # Sequence: user "ask", assistant "part-a", function_call, function_call_output, assistant "part-b"
        types_and_contents = [(p.type, p.role, p.content) for p in items]
        assert types_and_contents == [
            ("message", "user", "ask"),
            ("message", "assistant", "part-a"),
            ("function_call", None, None),
            ("function_call_output", None, None),
            ("message", "assistant", "part-b"),
        ]


class TestTurnInterleaving:
    def test_tool_items_emit_before_bot_message(self):
        c = conversation(msg("1", "ask"), bot_msg("2", "here"))
        historical_turns = {
            "2": turn(
                "2",
                function_call("call-a", "FileTool"),
                function_call_output("call-a", "ok"),
            )
        }
        items = project_for_llm(c, historical_turns)
        types = [p.type for p in items]
        assert types == [
            "message",
            "function_call",
            "function_call_output",
            "message",
        ]


class TestDeletedMessagesSkipped:
    def test_tombstoned_message_dropped(self):
        c = conversation(
            msg("1", "first"),
            msg("2", "removed", deleted=True),
            bot_msg("3", "bot reply"),
        )
        items = project_for_llm(c)
        contents = [p.content for p in items if p.role in {"user", "assistant"}]
        assert contents == ["first", "bot reply"]


class TestLiftedItems:
    def test_lifted_items_emit_before_anchor_bot_turn(self):
        """Lifted items are inserted immediately before the anchor bot turn's tool round —
        after any leading user prefix, adjacent to first file activity."""
        c = conversation(
            msg("1", "after-compaction question"),
            bot_msg("2", "anchor bot"),
            bot_msg("3", "later bot"),
            lifted_turn_items=[
                function_call("lifted-call", "FileTool", arguments='{"path": "/a"}'),
                function_call_output("lifted-call", "<file body>"),
            ],
            lifted_anchor_source_id="2",
        )
        historical_turns = {
            "2": turn(
                "2",
                function_call("turn-call", "FileTool"),
                function_call_output("turn-call", "ok"),
            ),
        }
        items = project_for_llm(c, historical_turns)
        types = [p.type for p in items]
        # user, lifted call+output, anchor's call+output, merged bot ("anchor bot\n\nlater bot")
        assert types == [
            "message",
            "function_call",
            "function_call_output",
            "function_call",
            "function_call_output",
            "message",
        ]
        assert items[-1].role == "assistant"
        assert items[-1].content == "anchor bot\n\nlater bot"
        # Lifted call leads; identified by its arguments
        call_args = [p.arguments for p in items if p.type == "function_call"]
        assert call_args[0] == '{"path": "/a"}'
        assert call_args[1] is None or call_args[1] == "{}"

    def test_lifted_items_skipped_when_anchor_is_later_bot(self):
        """Anchor at a later bot puts lifted items between earlier bot and anchor."""
        c = conversation(
            msg("1", "q"),
            bot_msg("2", "first bot"),
            msg("3", "q2"),
            bot_msg("4", "anchor bot"),
            lifted_turn_items=[
                function_call("lifted-call", "FileTool", arguments='{"path": "/a"}'),
                function_call_output("lifted-call", "<file body>"),
            ],
            lifted_anchor_source_id="4",
        )
        items = project_for_llm(c)
        types = [p.type for p in items]
        # Lifted items emit just before bot "4" — after the merged user "q\n\nq2" (since first bot has no turn, it
        # stays plain assistant).
        assert types == [
            "message",  # user q
            "message",  # first bot
            "message",  # user q2
            "function_call",
            "function_call_output",
            "message",  # anchor bot
        ]

    def test_no_lift_when_anchor_none(self):
        c = conversation(
            msg("1", "q"),
            bot_msg("2", "bot"),
            lifted_turn_items=[
                function_call("lifted-call", "FileTool"),
                function_call_output("lifted-call", "<body>"),
            ],
            lifted_anchor_source_id=None,
        )
        items = project_for_llm(c)
        # No insertion point → no lifted items emitted (will be cleaned up by compactor's invariant in practice, but the
        # projection itself stays defensive)
        assert all(p.type == "message" for p in items)
