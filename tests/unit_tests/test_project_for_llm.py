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


class TestAncestorSummaryProjection:
    def test_empty_summaries_produce_no_leading_summary_item(self):
        c = conversation(msg("1", "hi"), bot_msg("2", "hello"))
        items = project_for_llm(c)
        assert [(p.role, p.content) for p in items] == [
            ("user", "hi"),
            ("assistant", "hello"),
        ]

    def test_non_empty_summaries_lead_with_xml_block_when_first_message_is_bot(self):
        c = conversation(
            bot_msg("1", "first-assistant"),
            ancestor_summaries=["body"],
        )
        items = project_for_llm(c)
        assert items[0].role == "user"
        assert items[0].content is not None
        assert items[0].content.startswith('<compacted_summary trust="bot-summarized">\n')
        assert items[0].content.rstrip().endswith("</compacted_summary>")
        assert items[1].role == "assistant"

    def test_summary_merges_with_adjacent_user_first_message(self):
        c = conversation(
            msg("1", "real first user"),
            bot_msg("2", "bot reply"),
            ancestor_summaries=["body"],
        )
        items = project_for_llm(c)
        assert items[0].role == "user"
        assert items[0].content is not None
        assert items[0].content.startswith('<compacted_summary trust="bot-summarized">\n')
        assert items[0].content.endswith("\n\nreal first user")
        assert items[0].content.count("</compacted_summary>") == 1


class TestLeadingContextBlocks:
    def test_default_none_behaves_as_no_leading_blocks(self):
        c = conversation(msg("1", "hi"), bot_msg("2", "hello"))
        items = project_for_llm(c)
        assert [(p.role, p.content) for p in items] == [
            ("user", "hi"),
            ("assistant", "hello"),
        ]

    def test_empty_list_behaves_as_no_leading_blocks(self):
        c = conversation(msg("1", "hi"), bot_msg("2", "hello"))
        items = project_for_llm(c, leading_context_blocks=[])
        assert [(p.role, p.content) for p in items] == [
            ("user", "hi"),
            ("assistant", "hello"),
        ]

    def test_block_merges_after_summary_and_before_first_user(self):
        c = conversation(
            msg("1", "user-msg"),
            bot_msg("2", "bot-reply"),
            ancestor_summaries=["body"],
        )
        items = project_for_llm(
            c,
            leading_context_blocks=["<extra_block>X</extra_block>"],
        )
        assert items[0].role == "user"
        content = items[0].content or ""
        assert content.startswith('<compacted_summary trust="bot-summarized">\n')
        summary_end = content.index("</compacted_summary>") + len("</compacted_summary>")
        block_idx = content.index("<extra_block>X</extra_block>")
        user_idx = content.index("user-msg")
        assert summary_end < block_idx < user_idx

    def test_two_blocks_emit_in_list_order(self):
        c = conversation(
            msg("1", "user-msg"),
            bot_msg("2", "bot-reply"),
            ancestor_summaries=["body"],
        )
        items = project_for_llm(
            c,
            leading_context_blocks=["<block-a/>", "<block-b/>"],
        )
        content = items[0].content or ""
        a_idx = content.index("<block-a/>")
        b_idx = content.index("<block-b/>")
        user_idx = content.index("user-msg")
        assert a_idx < b_idx < user_idx

    def test_standalone_block_when_no_summary_and_no_adjacent_user(self):
        c = conversation(bot_msg("1", "bot-only"))
        items = project_for_llm(c, leading_context_blocks=["<solo/>"])
        assert [(p.role, p.content) for p in items] == [
            ("user", "<solo/>"),
            ("assistant", "bot-only"),
        ]

    def test_blocks_emit_as_user_role_messages_only(self):
        """Callers supply raw block content; projection wraps into user-role ProjectedItems. No path for an
        assistant or function item to enter via this parameter."""
        c = conversation(msg("1", "hi"))
        items = project_for_llm(c, leading_context_blocks=["<a/>", "<b/>"])
        leading = [p for p in items if (p.content or "").startswith("<")]
        assert leading
        assert all(p.type == "message" and p.role == "user" for p in leading)
