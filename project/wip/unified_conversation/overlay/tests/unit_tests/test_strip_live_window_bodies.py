"""strip_live_window_bodies: pre-summarization sanitization invariants.

Replaces every live-window `output` with a placeholder so the summarization
LLM call can't fossilize current file contents into `ancestor_summaries`.
Diagnostic headers (CONFLICT, ALREADY_EXISTS, RANGE_ERROR, RANGE_TRUNCATED)
are preserved because the model needs to react to them. Edit records,
tombstones, and non-live items pass through unchanged. The input list is
never mutated.
"""

from __future__ import annotations

from prokaryotes.conversation_v1.models import TurnItem
from prokaryotes.tools_v1.file_tool.live_windows import strip_live_window_bodies


def _live_window(
    *,
    call_id: str = "c1",
    path: str = "/tmp/x",
    output: str,
    status: str = "live",
) -> TurnItem:
    return TurnItem(
        call_id=call_id,
        type="function_call_output",
        output=output,
        prokaryotes_annotations={
            "file_tool.path": path,
            "file_tool.status": status,
            "file_tool.revision": "rev1",
            "file_tool.view_start_line": "1",
            "file_tool.view_end_line": "3",
        },
    )


def _function_call(call_id: str = "c1", name: str = "file_tool") -> TurnItem:
    return TurnItem(type="function_call", call_id=call_id, name=name, arguments="{}")


def test_strip_live_window_bodies_replaces_ordinary_live_window_wholesale():
    items = [_live_window(output="1 | line one\n2 | line two\n")]

    stripped = strip_live_window_bodies(items)

    assert stripped[0].output is not None
    assert "Live tracked file: /tmp/x" in stripped[0].output
    assert "line one" not in stripped[0].output
    assert "line two" not in stripped[0].output


def test_strip_live_window_bodies_preserves_conflict_diagnostic_header():
    """CONFLICT carries decision-relevant state — the diagnostic header before
    the `Current view` marker must survive; only the view body is stripped."""
    output = (
        "CONFLICT expected_revision=rev0 actual_revision=rev1\n"
        "Hint: re-read before retrying\n"
        "Current view: lines 1-3 of 10 (revision rev1)\n"
        "1 | a\n2 | b\n3 | c\n"
    )
    items = [_live_window(output=output)]

    stripped = strip_live_window_bodies(items)

    assert "CONFLICT expected_revision=rev0" in stripped[0].output
    assert "Hint: re-read before retrying" in stripped[0].output
    assert "Live tracked file" in stripped[0].output
    assert "1 | a" not in stripped[0].output


def test_strip_live_window_bodies_preserves_already_exists_diagnostic_header():
    output = "ALREADY_EXISTS path=/tmp/x\nCurrent view: lines 1-3 of 10 (revision rev1)\n1 | a\n2 | b\n3 | c\n"
    items = [_live_window(output=output)]

    stripped = strip_live_window_bodies(items)

    assert "ALREADY_EXISTS path=/tmp/x" in stripped[0].output
    assert "Live tracked file" in stripped[0].output
    assert "1 | a" not in stripped[0].output


def test_strip_live_window_bodies_preserves_range_error_diagnostic_header():
    output = "RANGE_ERROR requested 100-200 outside 1-10\nCurrent view: lines 1-3 of 10 (revision rev1)\n1 | a\n"
    items = [_live_window(output=output)]

    stripped = strip_live_window_bodies(items)

    assert "RANGE_ERROR requested 100-200" in stripped[0].output
    assert "Live tracked file" in stripped[0].output
    assert "1 | a" not in stripped[0].output


def test_strip_live_window_bodies_preserves_range_truncated_diagnostic_header():
    output = (
        "RANGE_TRUNCATED start=1 requested_end=10000 capped_end=200\n"
        "Current view: lines 1-3 of 10000 (revision rev1)\n"
        "1 | a\n"
    )
    items = [_live_window(output=output)]

    stripped = strip_live_window_bodies(items)

    assert "RANGE_TRUNCATED" in stripped[0].output
    assert "Live tracked file" in stripped[0].output
    assert "1 | a" not in stripped[0].output


def test_strip_live_window_bodies_handles_empty_current_view_marker():
    """When the diagnostic header is followed immediately by `Current view`
    with no diagnostic lines preceding it, the whole output collapses to the
    placeholder."""
    output = "Current view: lines 1-3 of 10 (revision rev1)\n1 | a\n2 | b\n"
    items = [_live_window(output=output)]

    stripped = strip_live_window_bodies(items)

    assert "Live tracked file" in stripped[0].output
    assert "1 | a" not in stripped[0].output


def test_strip_live_window_bodies_preserves_edit_records_and_tombstones():
    """Non-live function_call_output items (edit records, stale-marked
    tombstones) pass through unchanged. function_call items also pass through."""
    edit_record = TurnItem(
        call_id="c2",
        type="function_call_output",
        output="EDITED lines 1-2 with new content",
        prokaryotes_annotations={"file_tool.path": "/tmp/y", "file_tool.action": "edit"},
    )
    tombstone = _live_window(call_id="c3", output="STALE: file deleted", status="stale")
    fc = _function_call(call_id="c4", name="file_tool")
    items = [edit_record, tombstone, fc]

    stripped = strip_live_window_bodies(items)

    assert stripped[0].output == "EDITED lines 1-2 with new content"
    assert stripped[1].output == "STALE: file deleted"
    assert stripped[2] == fc


def test_strip_live_window_bodies_does_not_mutate_input_list():
    """Defensive: the returned list is a deep copy. Mutating items in the
    returned list must not bleed into the input."""
    original_output = "1 | original\n2 | content\n"
    items = [_live_window(output=original_output)]

    stripped = strip_live_window_bodies(items)
    stripped[0].output = "MUTATED"

    assert items[0].output == original_output


def test_strip_live_window_bodies_strips_inactive_paths_too():
    """The strip layer has no path-active filter — any live window gets
    stripped regardless of whether its path is currently in the active set.
    The lift planner handles active-path selection separately."""
    items = [
        _live_window(call_id="a", path="/tmp/active", output="1 | active body\n"),
        _live_window(call_id="b", path="/tmp/inactive", output="1 | inactive body\n"),
    ]

    stripped = strip_live_window_bodies(items)

    for s in stripped:
        assert "Live tracked file" in s.output
        assert "body" not in s.output  # bodies stripped from both
