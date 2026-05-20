"""Compaction's pre_tail-call_id filter: drops pre-tail-origin windows, keeps recency-tail and carryforward."""

from __future__ import annotations

from prokaryotes.context_v1.compaction import _file_tool_call_ids_in
from prokaryotes.conversation_v1.models import (
    TurnExecution,
    TurnItem,
    WorkingFileWindow,
)


def _file_call(call_id: str) -> TurnItem:
    return TurnItem(type="function_call", call_id=call_id, name="file_tool", arguments="{}")


def _output(call_id: str) -> TurnItem:
    return TurnItem(type="function_call_output", call_id=call_id, output="...")


def _turn(bot_id: str, *items: TurnItem) -> TurnExecution:
    return TurnExecution(conversation_uuid="c-1", bot_message_source_id=bot_id, items=list(items))


def _window(window_id: str) -> WorkingFileWindow:
    return WorkingFileWindow(
        window_id=window_id,
        path="/abs/a.py",
        status="live",
        revision="r1",
        rendered_output="FILE path=/abs/a.py revision=r1",
        view_start_line=1,
        view_end_line=40,
        requested_end_line=40,
        source_kind="read_lines",
    )


def _apply_pre_tail_filter(
    windows: list[WorkingFileWindow], pre_tail_turns: dict[str, TurnExecution]
) -> list[WorkingFileWindow]:
    """Mirror the carry-forward rule inside `_cas_swap_child` so we can test it in isolation."""
    pre_tail_call_ids = _file_tool_call_ids_in(pre_tail_turns)
    return [w for w in windows if w.window_id not in pre_tail_call_ids]


class TestPreTailFilter:
    def test_pre_tail_origin_window_drops(self):
        pre_tail_turns = {"b1": _turn("b1", _file_call("c-pre"), _output("c-pre"))}
        windows = [_window("c-pre"), _window("c-tail")]
        result = _apply_pre_tail_filter(windows, pre_tail_turns)
        assert [w.window_id for w in result] == ["c-tail"]

    def test_carryforward_window_survives_when_call_id_in_no_turns(self):
        """A window whose call_id is not in any pre_tail TurnExecution rides forward — this is the carryforward
        case (call_id originated in a compacted ancestor)."""
        pre_tail_turns = {"b1": _turn("b1", _file_call("c-pre"), _output("c-pre"))}
        windows = [_window("c-carry")]
        result = _apply_pre_tail_filter(windows, pre_tail_turns)
        assert [w.window_id for w in result] == ["c-carry"]

    def test_post_snapshot_window_survives_even_if_turn_execution_invisible(self):
        """Race-safety: a turn finalized during in-flight summarization writes Redis before its TurnExecution. At
        CAS time `current.working_file_windows` contains the post-snapshot window, but pre_tail_turns (loaded at
        compaction start) does NOT contain its call_id. The window survives."""
        pre_tail_turns = {"b1": _turn("b1", _file_call("c-pre"), _output("c-pre"))}
        # `c-post` was minted by a post-snapshot finalize_turn whose TurnExecution may or may not be in ES yet —
        # the filter doesn't read post-snapshot TurnExecutions, so it doesn't matter.
        windows = [_window("c-post")]
        result = _apply_pre_tail_filter(windows, pre_tail_turns)
        assert [w.window_id for w in result] == ["c-post"]

    def test_empty_pre_tail_keeps_everything(self):
        result = _apply_pre_tail_filter([_window("c-1"), _window("c-2")], {})
        assert [w.window_id for w in result] == ["c-1", "c-2"]
