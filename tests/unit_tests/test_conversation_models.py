from __future__ import annotations

from prokaryotes.conversation_v1.models import (
    compute_boundary_hash,
    compute_tail_hash,
    conversation_message_items,
)
from tests.unit_tests._builders import BOT_ID, bot_msg, conversation, msg


class TestSortedMessages:
    def test_orders_by_source_id_ascending(self):
        out_of_order = conversation(
            msg("1717000003.000000", "third"),
            msg("1717000001.000000", "first"),
            msg("1717000002.000000", "second"),
        )
        sorted_contents = [m.content for m in out_of_order.sorted_messages()]
        assert sorted_contents == ["first", "second", "third"]

    def test_does_not_mutate_storage_order(self):
        c = conversation(
            msg("1717000003.000000", "third"),
            msg("1717000001.000000", "first"),
        )
        c.sorted_messages()
        assert [m.content for m in c.messages] == ["third", "first"]


class TestConversationMessageItems:
    def test_filters_deleted(self):
        items = [
            msg("1", "alpha"),
            msg("2", "beta", deleted=True),
            msg("3", "gamma"),
        ]
        filtered = conversation_message_items(items)
        assert [m.content for m in filtered] == ["alpha", "gamma"]


class TestComputeBoundaryHash:
    def test_excludes_deleted(self):
        without_deleted = [msg("1", "alpha"), bot_msg("2", "beta")]
        with_deleted = [
            msg("1", "alpha"),
            msg("1.5", "dropped", deleted=True),
            bot_msg("2", "beta"),
        ]
        assert compute_boundary_hash(without_deleted) == compute_boundary_hash(with_deleted)

    def test_payload_includes_author_id_not_role(self):
        """Same content, different author → different hash. The payload keys on author_id (role is no longer
        stored)."""
        a = compute_boundary_hash([msg("1", "hi", author_id="u-alice")])
        b = compute_boundary_hash([msg("1", "hi", author_id="u-bob")])
        assert a != b


class TestComputeTailHash:
    def test_uses_only_non_bot_messages(self):
        items = [
            msg("1", "alpha", author_id="u-alice"),
            bot_msg("2", "beta"),
            msg("3", "gamma", author_id="u-alice"),
        ]
        bot_only = [bot_msg("2", "beta")]
        non_bot_only = [
            msg("1", "alpha", author_id="u-alice"),
            msg("3", "gamma", author_id="u-alice"),
        ]
        assert compute_tail_hash(items, BOT_ID) == compute_tail_hash(non_bot_only, BOT_ID)
        assert compute_tail_hash(items, BOT_ID) != compute_tail_hash(bot_only, BOT_ID)

    def test_takes_last_n(self):
        items = [msg(str(i), f"content-{i}", author_id="u-alice") for i in range(10)]
        h_last_2 = compute_tail_hash(items, BOT_ID, n=2)
        h_last_2_explicit = compute_tail_hash(items[-2:], BOT_ID, n=2)
        assert h_last_2 == h_last_2_explicit

    def test_multi_author_treated_uniformly(self):
        """`compute_tail_hash` takes content of any non-bot author. Two authors at the same source_id sequence
        with the same contents produce the same tail hash."""
        with_alice_and_bob = [
            msg("1", "alpha", author_id="u-alice"),
            msg("2", "beta", author_id="u-bob"),
        ]
        with_alice_only = [
            msg("1", "alpha", author_id="u-alice"),
            msg("2", "beta", author_id="u-alice"),
        ]
        assert compute_tail_hash(with_alice_and_bob, BOT_ID) == compute_tail_hash(with_alice_only, BOT_ID)


class TestAncestorSummaryBlock:
    def test_none_when_no_summaries(self):
        c = conversation(msg("1", "hi"))
        assert c.ancestor_summary_block() is None

    def test_joins_summaries_with_blank_lines(self):
        c = conversation(msg("1", "hi"), ancestor_summaries=["summary-1", "summary-2"])
        block = c.ancestor_summary_block()
        assert block is not None
        assert "summary-1" in block
        assert "summary-2" in block
        assert "summary-1\n\nsummary-2" in block

    def test_single_summary_emits_opening_and_closing_tags(self):
        c = conversation(msg("1", "hi"), ancestor_summaries=["body-one"])
        block = c.ancestor_summary_block()
        assert block is not None
        assert block.startswith('<compacted_summary trust="bot-summarized">\n')
        assert block.endswith("\n</compacted_summary>")
        assert "body-one" in block

    def test_header_sentences_precede_bodies(self):
        c = conversation(msg("1", "hi"), ancestor_summaries=["body-one"])
        block = c.ancestor_summary_block()
        assert block is not None
        header_anchor = "The following is background memory summarizing earlier turns in this conversation."
        assert header_anchor in block
        assert block.index(header_anchor) < block.index("body-one")

    def test_multiple_summaries_sit_inside_closing_tag(self):
        c = conversation(msg("1", "hi"), ancestor_summaries=["body-one", "body-two"])
        block = c.ancestor_summary_block()
        assert block is not None
        body_section = block.split("\n\n", 1)[1]
        assert body_section.endswith("body-one\n\nbody-two\n</compacted_summary>")


class TestAncestorSummaryBlockClosingTagEscape:
    def test_closing_tag_in_body_is_escaped(self):
        c = conversation(msg("1", "hi"), ancestor_summaries=["pre </compacted_summary> post"])
        block = c.ancestor_summary_block()
        assert block is not None
        assert "<\\/compacted_summary>" in block
        # The only un-escaped closing tag is the structural one at the very end
        assert block.count("</compacted_summary>") == 1
        assert block.rstrip().endswith("</compacted_summary>")

    def test_opening_tag_substring_in_body_is_left_alone(self):
        c = conversation(
            msg("1", "hi"),
            ancestor_summaries=['quote: <compacted_summary trust="something-else"> stays as-is'],
        )
        block = c.ancestor_summary_block()
        assert block is not None
        assert block.count('<compacted_summary trust="bot-summarized">') == 1
        assert '<compacted_summary trust="something-else">' in block

    def test_escape_applies_per_summary_independently(self):
        c = conversation(
            msg("1", "hi"),
            ancestor_summaries=["clean body", "leaky </compacted_summary> body"],
        )
        block = c.ancestor_summary_block()
        assert block is not None
        assert "clean body" in block
        assert "leaky <\\/compacted_summary> body" in block
        assert block.count("</compacted_summary>") == 1
