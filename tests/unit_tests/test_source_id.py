"""Unit tests for `conversation_v1.source_id` — the single source of truth for the `ConversationMessage.source_id`
invariant (format, monotonic bump, sorted insert).
"""

from __future__ import annotations

import pytest

from prokaryotes.conversation_v1.source_id import (
    bump_source_id,
    format_source_id,
    format_source_id_now,
    insert_message_sorted,
)
from tests.unit_tests._builders import msg


class TestBumpSourceId:
    def test_carries_seconds_at_microsecond_overflow(self):
        assert bump_source_id("1717000000.999999") == "1717000001.000000"

    def test_handles_micros_missing_after_partition(self):
        """A bare seconds string with no `.` is parseable: the partition yields an empty micros half, which the
        helper coerces to 0 before bumping."""
        assert bump_source_id("1717000000") == "1717000000.000001"

    def test_increments_microseconds(self):
        assert bump_source_id("1717000000.000000") == "1717000000.000001"
        assert bump_source_id("1717000000.123456") == "1717000000.123457"

    @pytest.mark.parametrize(
        "bad_input",
        [
            "not-a-source-id",
            "abc.def",
            "1717000000.notmicros",
            "",
            ".",
            "1.2.3",
        ],
    )
    def test_invalid_input_falls_back_to_wall_clock(self, bad_input: str):
        """Malformed input never raises — fallback returns a fresh wall-clock `source_id`."""
        out = bump_source_id(bad_input)
        # Wall-clock output is a parseable `source_id` (otherwise it could itself be re-bumped wrong).
        assert "." in out
        seconds_str, _, micros_str = out.partition(".")
        assert seconds_str.isdigit() and micros_str.isdigit()
        assert len(micros_str) == 6


class TestFormatSourceId:
    def test_lex_sort_equals_chronological_for_realistic_timestamps(self):
        """With fixed-width 10-digit unix seconds, lexicographic sort equals chronological sort — what makes
        `source_id` viable as identity + ordering."""
        a = format_source_id(1717000000.123456)
        b = format_source_id(1717000001.000000)
        c = format_source_id(1717000999.999999)
        assert a < b < c

    def test_seconds_microseconds_format(self):
        assert format_source_id(1717000000.123456) == "1717000000.123456"

    def test_zero_pads_microseconds(self):
        assert format_source_id(1717000000.0) == "1717000000.000000"


class TestFormatSourceIdNow:
    def test_returns_parseable_source_id(self):
        out = format_source_id_now()
        seconds_str, _, micros_str = out.partition(".")
        assert seconds_str.isdigit() and micros_str.isdigit()
        assert len(micros_str) == 6


class TestInsertMessageSorted:
    def test_inserts_into_empty_list(self):
        messages = []
        insert_message_sorted(messages, msg("100.000000", "x"))
        assert [m.source_id for m in messages] == ["100.000000"]

    def test_inserts_into_middle(self):
        messages = [msg("100.000000", "a"), msg("300.000000", "c")]
        insert_message_sorted(messages, msg("200.000000", "b"))
        assert [m.source_id for m in messages] == ["100.000000", "200.000000", "300.000000"]

    def test_out_of_order_insert_restores_sort_order(self):
        """The whole point of the helper: a same-thread out-of-order delivery should land in its sorted position,
        not at the tail."""
        messages = [msg("100.000000", "a"), msg("300.000000", "c")]
        insert_message_sorted(messages, msg("200.000000", "b"))
        # Followed by an even-earlier source_id arriving late — still finds its sorted spot.
        insert_message_sorted(messages, msg("050.000000", "earlier"))
        assert [m.source_id for m in messages] == [
            "050.000000",
            "100.000000",
            "200.000000",
            "300.000000",
        ]

    def test_tail_append_when_message_is_newest(self):
        messages = [msg("100.000000", "a"), msg("200.000000", "b")]
        insert_message_sorted(messages, msg("300.000000", "c"))
        assert [m.source_id for m in messages] == ["100.000000", "200.000000", "300.000000"]

    def test_tie_breaks_after_equal_source_id(self):
        """`bisect_right` places ties after existing equal entries — preserves arrival order for same-ts messages."""
        messages = [msg("100.000000", "first")]
        insert_message_sorted(messages, msg("100.000000", "second"))
        assert [m.content for m in messages] == ["first", "second"]
