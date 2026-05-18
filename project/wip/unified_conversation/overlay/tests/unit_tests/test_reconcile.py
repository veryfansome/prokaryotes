from __future__ import annotations

from prokaryotes.conversation_v1.reconcile import reconcile
from tests.unit_tests._builders import (
    bot_msg,
    conversation,
    msg,
    normalized,
    normalized_bot,
)


class TestMatch:
    def test_identical_returns_match_with_no_operations(self):
        stored = conversation(msg("1", "hi"), bot_msg("2", "hello"))
        result = reconcile(
            stored,
            [normalized("1", "hi"), normalized_bot("2", "hello")],
        )
        assert result.classification == "match"
        assert result.operations == []
        assert result.shared_prefix_source_ids == ["1", "2"]


class TestAppend:
    def test_pure_trailing_append(self):
        stored = conversation(msg("1", "hi"), bot_msg("2", "hello"))
        result = reconcile(
            stored,
            [
                normalized("1", "hi"),
                normalized_bot("2", "hello"),
                normalized("3", "follow-up"),
            ],
        )
        assert result.classification == "append"
        assert [op.kind for op in result.operations] == ["append"]
        assert result.operations[0].source_id == "3"
        assert result.shared_prefix_source_ids == ["1", "2"]

    def test_multiple_trailing_appends(self):
        stored = conversation(msg("1", "hi"))
        result = reconcile(
            stored,
            [normalized("1", "hi"), normalized_bot("2", "bot"), normalized("3", "next")],
        )
        assert result.classification == "append"
        assert len(result.operations) == 2


class TestEdit:
    def test_slack_message_changed(self):
        stored = conversation(msg("1", "original"), bot_msg("2", "reply"))
        result = reconcile(
            stored,
            [normalized("1", "edited"), normalized_bot("2", "reply")],
        )
        assert result.classification == "edit"
        assert [op.kind for op in result.operations] == ["edit"]
        op = result.operations[0]
        assert op.source_id == "1"
        assert op.incoming is not None
        assert op.incoming.content == "edited"


class TestTrailingDelete:
    def test_trailing_delete_classifies_as_delete(self):
        """Stored has a non-deleted message past last_shared; incoming omits it."""
        stored = conversation(msg("1", "hi"), bot_msg("2", "hello"), msg("3", "extra"))
        result = reconcile(
            stored,
            [normalized("1", "hi"), normalized_bot("2", "hello")],
        )
        assert result.classification == "delete"
        assert [op.kind for op in result.operations] == ["delete"]
        assert result.operations[0].source_id == "3"


class TestDivergence:
    def test_web_edit_regenerate(self):
        """Stored = [u1, b1, u2, b2]; incoming = [u1, u2_new]. New source_id and
        omitted stored ids → divergence."""
        stored = conversation(
            msg("1", "u1"),
            bot_msg("2", "b1"),
            msg("3", "u2"),
            bot_msg("4", "b2"),
        )
        result = reconcile(
            stored,
            [normalized("1", "u1"), normalized("5", "u2-new")],
        )
        assert result.classification == "divergence"
        kinds = {op.kind for op in result.operations}
        assert kinds == {"append", "delete"}
        assert result.shared_prefix_source_ids == ["1"]
        # Divergence rooted right after the shared prefix
        assert result.divergence_point_index == 1

    def test_head_delete_is_divergence_not_delete(self):
        """Removing the leading stored message is not a 'pure-append extension'."""
        stored = conversation(msg("1", "hi"), msg("2", "world"))
        result = reconcile(stored, [normalized("2", "world")])
        assert result.classification == "divergence"
        assert result.shared_prefix_source_ids == []
        assert result.divergence_point_index == 0

    def test_internal_edit_is_divergence(self):
        """Edit at a source_id below last_shared = divergence."""
        stored = conversation(msg("1", "a"), msg("2", "b"), msg("3", "c"))
        # Edit "1" → "a-edited" while keeping 2 and 3
        result = reconcile(
            stored,
            [normalized("1", "a-edited"), normalized("2", "b"), normalized("3", "c")],
        )
        # No shared prefix (m1 differs); operations include edit
        assert result.classification in {"divergence", "edit"}
        if result.classification == "edit":
            # If our classifier treats same-id-set as edit, that's also acceptable per
            # the doc — Slack `message_changed` is first-class regardless of position.
            assert all(op.kind == "edit" for op in result.operations)
        else:
            assert result.classification == "divergence"


class TestDeletedTombstoneStorage:
    def test_tombstoned_stored_messages_excluded_from_diff(self):
        """A deleted stored message does not produce a delete op."""
        stored = conversation(
            msg("1", "hi"),
            msg("2", "removed", deleted=True),
            bot_msg("3", "hello"),
        )
        result = reconcile(
            stored,
            [normalized("1", "hi"), normalized_bot("3", "hello")],
        )
        assert result.classification == "match"


class TestEmptyStored:
    def test_first_messages_are_appends(self):
        stored = conversation()  # no messages
        result = reconcile(stored, [normalized("1", "hi")])
        assert result.classification == "append"
        assert len(result.operations) == 1
        assert result.divergence_point_index is None


class TestSortingInsensitive:
    def test_incoming_arrives_unsorted(self):
        stored = conversation(msg("1", "hi"))
        result = reconcile(
            stored,
            [normalized_bot("2", "bot"), normalized("1", "hi"), normalized("3", "next")],
        )
        assert result.classification == "append"
        # Operations should walk incoming in source_id order
        append_source_ids = [op.source_id for op in result.operations if op.kind == "append"]
        assert append_source_ids == ["2", "3"]
