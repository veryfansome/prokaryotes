"""Tests for FileTool window deduplication.

Covers the pure interval engine (`consolidate_intervals`), the redesigned `_do_read_lines` read flow
(coverage, provenance, empty/past-EOF, consolidation, mid-turn growth), the reconcile fold, and the
`WorkingFileWindow` schema invariants. See `project/features/file_tool/README.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from prokaryotes.context_v1.compaction import (
    _carry_forward_windows,
)
from prokaryotes.context_v1.compaction import (
    _file_tool_call_ids_in as _compaction_call_ids_in,
)
from prokaryotes.context_v1.conversation_sync import (
    _active_paths_in_turns,
    _file_tool_call_ids_in,
    _filter_windows_by_active_path_and_origin,
)
from prokaryotes.conversation_v1.models import (
    Conversation,
    TurnExecution,
    TurnItem,
    WorkingFileWindow,
    compute_boundary_hash,
    compute_tail_hash,
)
from prokaryotes.tools_v1.file_tool import FileTool, live_windows
from prokaryotes.tools_v1.file_tool.intervals import Interval, consolidate_intervals
from tests.unit_tests._builders import BOT_ID, bot_msg, msg
from tests.unit_tests._fakes import make_syncer

# --------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------


def _arguments(action: str, path: str, **extra) -> str:
    return json.dumps(
        {
            "action": action,
            "path": path,
            "expected_revision": extra.get("expected_revision"),
            "start_line": extra.get("start_line"),
            "end_line": extra.get("end_line"),
            "new_text": extra.get("new_text"),
        }
    )


def _write_lines(path: Path, n: int) -> None:
    path.write_text("\n".join(f"line-{i}" for i in range(1, n + 1)) + "\n", encoding="utf-8")


def _write_marked(path: Path, n: int, marker: str) -> None:
    path.write_text("\n".join(f"{marker}-{i}" for i in range(1, n + 1)) + "\n", encoding="utf-8")


def _win(path: str, start: int, end: int, *, origins: list[str], window_id: str, line_count: int) -> WorkingFileWindow:
    return WorkingFileWindow(
        window_id=window_id,
        path=path,
        status="live",
        revision="r",
        rendered_output="",
        view_start_line=start,
        view_end_line=end,
        requested_end_line=end,
        line_count=line_count,
        origin_call_ids=origins,
        source_kind="read_lines",
    )


def _windows_for(tool_windows: list[WorkingFileWindow], path: Path) -> list[WorkingFileWindow]:
    return [w for w in tool_windows if w.path == str(path)]


# --------------------------------------------------------------------------------------------------------------
# Interval / consolidate_intervals (pure)
# --------------------------------------------------------------------------------------------------------------


def test_interval_predicates():
    assert Interval(1, 10).contains(Interval(2, 5))
    assert not Interval(1, 10).contains(Interval(2, 11))
    assert Interval(1, 10).overlaps(Interval(10, 20))
    assert not Interval(1, 10).overlaps(Interval(11, 20))
    assert Interval(1, 10).touches_or_overlaps(Interval(11, 20))  # adjacent
    assert not Interval(1, 10).touches_or_overlaps(Interval(12, 20))  # gap
    assert Interval(1, 10).union(Interval(11, 20)) == Interval(1, 20)
    assert Interval(1, 10).union(Interval(12, 20)) is None


def test_consolidate_none():
    result = consolidate_intervals([], Interval(1, 50), 200)
    assert result.primary == Interval(1, 50)
    assert result.secondaries == []
    assert result.retired == []
    assert result.unreached == []


def test_consolidate_superset():
    result = consolidate_intervals([Interval(10, 20)], Interval(1, 50), 200)
    assert result.primary == Interval(1, 50)
    assert result.retired == [Interval(10, 20)]


def test_consolidate_partial_overlap_fits():
    result = consolidate_intervals([Interval(1, 60)], Interval(30, 80), 200)
    assert result.primary == Interval(1, 80)
    assert result.retired == [Interval(1, 60)]
    assert result.secondaries == []


def test_consolidate_partial_overlap_splits():
    result = consolidate_intervals([Interval(1, 80)], Interval(50, 140), 100)
    assert result.primary == Interval(50, 140)  # anchored on new
    assert result.secondaries == [Interval(1, 49)]
    assert result.retired == [Interval(1, 80)]


def test_consolidate_contiguous_touch_merges():
    result = consolidate_intervals([Interval(1, 50)], Interval(51, 100), 200)
    assert result.primary == Interval(1, 100)
    assert result.retired == [Interval(1, 50)]


def test_consolidate_page_through_splits():
    result = consolidate_intervals([Interval(1, 200)], Interval(201, 400), 200)
    assert result.primary == Interval(201, 400)
    assert result.secondaries == [Interval(1, 200)]
    assert result.retired == [Interval(1, 200)]


def test_consolidate_disjoint_untouched():
    result = consolidate_intervals([Interval(1, 50)], Interval(100, 150), 200)
    assert result.primary == Interval(100, 150)
    assert result.retired == []
    assert result.unreached == [Interval(1, 50)]


def test_consolidate_chained_reachability():
    result = consolidate_intervals([Interval(1, 50), Interval(51, 100)], Interval(40, 60), 200)
    assert result.primary == Interval(1, 100)
    assert set((i.start, i.end) for i in result.retired) == {(1, 50), (51, 100)}


# --------------------------------------------------------------------------------------------------------------
# _do_read_lines read flow
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_mints_window_with_new_fields(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 3)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=3), call_id="c-1")
    assert len(windows) == 1
    w = windows[0]
    assert w.window_id == "c-1"
    assert w.line_count == 3
    assert w.origin_call_ids == ["c-1"]
    assert w.requested_end_line == 3
    assert w.source_kind == "read_lines"
    assert w.view_start_line == 1 and w.view_end_line == 3


@pytest.mark.asyncio
async def test_redundant_subset_appends_provenance(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 10)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="c-1")
    second = await tool.call(_arguments("read_lines", str(target), start_line=2, end_line=5), call_id="c-2")
    assert (second.output or "").startswith("REDUNDANT_READ")
    assert "Use that window" in (second.output or "")
    assert len(windows) == 1
    assert windows[0].origin_call_ids == ["c-1", "c-2"]


@pytest.mark.asyncio
async def test_redundant_eof_message(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 10)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="c-1")
    second = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=50), call_id="c-2")
    assert (second.output or "").startswith("REDUNDANT_READ")
    assert "File ends at line 10" in (second.output or "")


@pytest.mark.asyncio
async def test_redundant_cap_message(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 400)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=200), call_id="c-1")
    second = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=300), call_id="c-2")
    assert (second.output or "").startswith("REDUNDANT_READ")
    assert "page forward from start_line=201" in (second.output or "")


@pytest.mark.asyncio
async def test_redundant_cap_wins_over_eof_when_cap_below_line_count(tmp_path: Path):
    # cap (200) < line_count (250) < request.end (300): the (1,200) window stopped at the per-call cap with lines
    # 201-250 still UNREAD. The redundant guidance must tell the model to page forward from 201, NOT "file ends /
    # do not page forward" — the EOF message is correct only when the window itself reaches EOF. Guards the bug
    # where the EOF branch fired on request.end > line_count regardless of whether the window covered up to EOF.
    target = tmp_path / "a.txt"
    _write_lines(target, 250)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=200), call_id="c-1")
    second = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=300), call_id="c-2")
    out = second.output or ""
    assert out.startswith("REDUNDANT_READ"), out
    assert "page forward from start_line=201" in out, out
    assert "File ends at line" not in out, out  # content remains past the window — EOF guidance would be wrong


@pytest.mark.asyncio
async def test_page_through_two_contiguous_windows(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 400)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=200), call_id="c-1")
    second = await tool.call(_arguments("read_lines", str(target), start_line=201, end_line=400), call_id="c-2")
    assert (second.output or "").startswith("FILE ")
    assert len(windows) == 2
    by_start = {w.view_start_line: w for w in windows}
    assert set(by_start) == {1, 201}
    assert by_start[201].window_id == "c-2"
    assert by_start[1].window_id.startswith("wfw-")
    # both share the union of origins
    assert by_start[1].origin_call_ids == ["c-1", "c-2"]
    assert by_start[201].origin_call_ids == ["c-1", "c-2"]


@pytest.mark.asyncio
async def test_symlink_trio_single_window(tmp_path: Path):
    target = tmp_path / "README.md"
    _write_lines(target, 3)
    (tmp_path / "AGENTS.md").symlink_to(target)
    (tmp_path / "CLAUDE.md").symlink_to(target)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    for i, name in enumerate(["README.md", "AGENTS.md", "CLAUDE.md"], start=1):
        await tool.call(_arguments("read_lines", str(tmp_path / name), start_line=1, end_line=3), call_id=f"c-{i}")
    assert len(windows) == 1


@pytest.mark.asyncio
async def test_empty_file_placeholder(tmp_path: Path):
    target = tmp_path / "empty.txt"
    target.write_text("", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    first = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="c-1")
    assert "line_count=0" in (first.output or "")
    assert len(windows) == 1
    assert windows[0].view_start_line == 1 and windows[0].view_end_line == 0
    assert windows[0].line_count == 0
    # a repeat read is covered by the placeholder
    second = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="c-2")
    assert (second.output or "").startswith("REDUNDANT_READ")
    assert len(windows) == 1


@pytest.mark.asyncio
async def test_past_eof_mints_empty_view_others_untouched(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 5)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    # window that does NOT reach EOF, so the past-EOF read isn't covered
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=3), call_id="c-1")
    await tool.call(_arguments("read_lines", str(target), start_line=50, end_line=60), call_id="c-2")
    assert len(windows) == 2
    placeholder = next(w for w in windows if w.window_id == "c-2")
    assert placeholder.view_start_line == 50 and placeholder.view_end_line == 49
    assert placeholder.line_count == 5
    assert any(w.window_id == "c-1" and w.view_end_line == 3 for w in windows)


@pytest.mark.asyncio
async def test_diagnostic_does_not_grant_redundant_then_folds(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 10)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    # create_file on an existing file mints an ALREADY_EXISTS diagnostic window
    created = await tool.call(_arguments("create_file", str(target), new_text="x\n"), call_id="w-1")
    assert (created.output or "").startswith("ALREADY_EXISTS")
    assert windows[0].source_kind == "already_exists"
    # a read whose range the diagnostic covers must NOT short-circuit; it folds the diagnostic
    read = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="r-1")
    assert (read.output or "").startswith("FILE ")
    assert len(windows) == 1
    w = windows[0]
    assert w.window_id == "r-1"
    assert w.source_kind == "read_lines"
    assert w.origin_call_ids == ["r-1", "w-1"]


@pytest.mark.asyncio
async def test_midturn_inplace_change_reread_renders_current_not_redundant(tmp_path: Path):
    # Same line count, different content (a revision change the line-extent coverage check can't see). A same-turn
    # re-read must surface the new content, not a REDUNDANT_READ pointer the model can't resolve to fresh text.
    target = tmp_path / "a.txt"
    target.write_text("a\nb\nc\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    first = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=3), call_id="c-1")
    assert "a" in (first.output or "")
    target.write_text("X\nY\nZ\n", encoding="utf-8")  # same 3 lines, new content -> new revision
    second = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=3), call_id="c-2")
    assert (second.output or "").startswith("FILE ")
    assert "REDUNDANT" not in (second.output or "")
    assert "X" in (second.output or "")  # current content surfaced
    # the durable window is re-minted at the new content and stays a single non-overlapping window
    assert len(windows) == 1
    assert windows[0].view_start_line == 1 and windows[0].view_end_line == 3


@pytest.mark.asyncio
async def test_disjoint_read_then_reread_renders_current(tmp_path: Path):
    # read 1-3 -> external whole-file change (same line count) -> disjoint read 8-10 (silently refreshes the 1-3
    # window durably) -> re-read 1-3. The re-read must surface the new content, not REDUNDANT_READ from a window
    # the model was never shown at the new revision.
    target = tmp_path / "a.txt"
    target.write_text("a\nb\nc\nd\ne\nf\ng\nh\ni\nj\n", encoding="utf-8")  # 10 lines, R1
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    first = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=3), call_id="c-1")
    assert "a" in (first.output or "")
    target.write_text("A\nB\nC\nd\ne\nf\ng\nH\nI\nJ\n", encoding="utf-8")  # same 10 lines, new content, R2
    disjoint = await tool.call(_arguments("read_lines", str(target), start_line=8, end_line=10), call_id="c-2")
    assert (disjoint.output or "").startswith("FILE ")  # disjoint read refreshes the 1-3 window as a side effect
    third = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=3), call_id="c-3")
    assert (third.output or "").startswith("FILE ")
    assert "REDUNDANT" not in (third.output or "")
    assert "A" in (third.output or "")  # new lines 1-3 surfaced, not the stale "a"


@pytest.mark.asyncio
async def test_disjoint_read_leaves_no_mixed_revisions(tmp_path: Path):
    # refresh-then-consolidate currency: a read refreshes ALL live windows for the path (not just the one it
    # consolidates) before deciding anything, so a disjoint read can never leave two windows for a path at mixed
    # revisions mid-turn — the stale sibling is brought current as a side effect.
    target = tmp_path / "a.txt"
    _write_marked(target, 20, "old")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=3), call_id="c-1")
    _write_marked(target, 20, "new")  # whole-file change mid-turn -> new revision
    await tool.call(_arguments("read_lines", str(target), start_line=10, end_line=12), call_id="c-2")
    path_windows = _windows_for(windows, target)
    assert len(path_windows) == 2  # (1,3) and (10,12): disjoint, non-touching -> both kept
    assert len({w.revision for w in path_windows}) == 1  # sibling (1,3) refreshed to the new revision too


@pytest.mark.asyncio
async def test_consolidation_extended_primary_unshown_tail_renders(tmp_path: Path):
    # read 1-80 -> external whole-file change -> read 50-140 consolidates to primary (1,140) but only renders
    # 50-140. Lines 1-49 at the new revision were never shown, so a re-read must render them, not REDUNDANT.
    target = tmp_path / "a.txt"
    _write_marked(target, 140, "old")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=80), call_id="c-1")
    _write_marked(target, 140, "new")  # whole file changed -> new revision
    second = await tool.call(_arguments("read_lines", str(target), start_line=50, end_line=140), call_id="c-2")
    assert (second.output or "").startswith("FILE ")
    third = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=49), call_id="c-3")
    assert (third.output or "").startswith("FILE ")
    assert "REDUNDANT" not in (third.output or "")
    assert "new-1" in (third.output or "")  # unshown tail surfaced at the new revision


@pytest.mark.asyncio
async def test_consolidation_secondary_unshown_renders(tmp_path: Path):
    # read 1-200 -> external whole-file change -> read 201-400 splits off a secondary (1,200) that was never
    # rendered. A re-read of 1-200 must render the new content, not REDUNDANT off the unshown secondary.
    target = tmp_path / "a.txt"
    _write_marked(target, 400, "old")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=200), call_id="c-1")
    _write_marked(target, 400, "new")
    second = await tool.call(_arguments("read_lines", str(target), start_line=201, end_line=400), call_id="c-2")
    assert (second.output or "").startswith("FILE ")
    third = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=200), call_id="c-3")
    assert (third.output or "").startswith("FILE ")
    assert "REDUNDANT" not in (third.output or "")
    assert "new-1" in (third.output or "")


@pytest.mark.asyncio
async def test_unchanged_reread_still_redundant(tmp_path: Path):
    # Guard against over-firing: with no content change, a covered re-read is still REDUNDANT_READ.
    target = tmp_path / "a.txt"
    _write_lines(target, 10)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="c-1")
    second = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="c-2")
    assert (second.output or "").startswith("REDUNDANT_READ")
    assert len(windows) == 1


@pytest.mark.asyncio
async def test_coverage_after_refresh_under_midturn_growth(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 50)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=50), call_id="c-1")
    # external mid-turn growth
    _write_lines(target, 80)
    second = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=100), call_id="c-2")
    assert (second.output or "").startswith("FILE ")  # NOT redundant — the new lines surface
    assert "line_count=80" in (second.output or "")
    assert len(windows) == 1
    assert windows[0].view_start_line == 1 and windows[0].view_end_line == 80


# --------------------------------------------------------------------------------------------------------------
# reconcile fold (live_windows.fold_windows_for_path)
# --------------------------------------------------------------------------------------------------------------


def test_fold_collapses_overlap():
    text = "\n".join(f"line-{i}" for i in range(1, 101)) + "\n"
    path = "/p/a.txt"
    windows = [
        _win(path, 1, 100, origins=["a"], window_id="a", line_count=100),
        _win(path, 50, 60, origins=["b"], window_id="b", line_count=100),
    ]
    live_windows.fold_windows_for_path(windows, path, text, "rev", 200)
    assert len(windows) == 1
    assert (windows[0].view_start_line, windows[0].view_end_line) == (1, 100)
    assert windows[0].origin_call_ids == ["a", "b"]
    assert windows[0].window_id.startswith("wfw-")


def test_fold_noop_preserves_identity():
    text = "\n".join(f"line-{i}" for i in range(1, 101)) + "\n"
    path = "/p/a.txt"
    windows = [
        _win(path, 1, 50, origins=["a"], window_id="a", line_count=100),
        _win(path, 51, 100, origins=["b"], window_id="b", line_count=100),  # contiguous, not overlapping
    ]
    live_windows.fold_windows_for_path(windows, path, text, "rev", 200)
    assert len(windows) == 2
    assert {w.window_id for w in windows} == {"a", "b"}


# --------------------------------------------------------------------------------------------------------------
# WorkingFileWindow schema invariants
# --------------------------------------------------------------------------------------------------------------


def test_origin_call_ids_validator_sorts_and_dedups():
    w = _win("/p/a.txt", 1, 10, origins=["b", "a", "b"], window_id="x", line_count=10)
    assert w.origin_call_ids == ["a", "b"]


def test_origin_call_ids_validator_rejects_empty():
    with pytest.raises(ValidationError):
        _win("/p/a.txt", 1, 10, origins=[], window_id="x", line_count=10)


def test_requested_end_line_is_required():
    with pytest.raises(ValidationError):
        WorkingFileWindow(
            window_id="x",
            path="/p/a.txt",
            status="live",
            rendered_output="",
            view_start_line=1,
            view_end_line=10,
            line_count=10,
            origin_call_ids=["x"],
            source_kind="read_lines",
        )


# --------------------------------------------------------------------------------------------------------------
# working_files_block projects in monotonic (path, view_start_line) order regardless of storage order
# (consolidation mints primary-before-lower-secondary; the reconcile fold appends re-minted windows at the end).
# --------------------------------------------------------------------------------------------------------------


def _block_index(block: str, needle: str) -> int:
    idx = (block or "").find(needle)
    assert idx >= 0, f"{needle!r} not found in block:\n{block}"
    return idx


def test_working_files_block_projects_in_line_order():
    # Storage order is scrambled across two paths; the projected block must read sorted by (path, start).
    w_a_hi = _win("/p/a.txt", 200, 250, origins=["c1"], window_id="c1", line_count=300)
    w_b = _win("/p/b.txt", 1, 10, origins=["c2"], window_id="c2", line_count=10)
    w_a_lo = _win("/p/a.txt", 1, 50, origins=["c3"], window_id="c3", line_count=300)
    conv = Conversation(
        conversation_uuid="u",
        bot_author_id="bot",
        working_file_windows=[w_a_hi, w_b, w_a_lo],  # deliberately out of order
    )
    block = conv.working_files_block() or ""
    # a.txt before b.txt; within a.txt, lines 1-50 before 200-250.
    assert _block_index(block, "/p/a.txt lines 1-50") < _block_index(block, "/p/a.txt lines 200-250")
    assert _block_index(block, "/p/a.txt lines 200-250") < _block_index(block, "/p/b.txt lines 1-10")


@pytest.mark.asyncio
async def test_page_through_split_projects_in_line_order(tmp_path: Path):
    # Real flow: page-through stores [primary (201,400), secondary (1,200)] — non-monotonic on the list.
    # The projection must still read (1,200) before (201,400).
    target = tmp_path / "a.txt"
    _write_lines(target, 400)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=200), call_id="c-1")
    await tool.call(_arguments("read_lines", str(target), start_line=201, end_line=400), call_id="c-2")
    assert [w.view_start_line for w in windows] == [201, 1]  # storage order is non-monotonic
    conv = Conversation(conversation_uuid="u", bot_author_id="bot", working_file_windows=windows)
    block = conv.working_files_block() or ""
    assert _block_index(block, "lines 1-200") < _block_index(block, "lines 201-400")


# --------------------------------------------------------------------------------------------------------------
# branch / cold-rebuild origin filter (the multi-origin quantifier fix)
# --------------------------------------------------------------------------------------------------------------


_P = "/p/a.txt"


def _filter(windows, *, kept, source):
    return _filter_windows_by_active_path_and_origin(
        windows, active_paths={_P}, kept_call_ids=kept, source_call_ids=source
    )


def test_filter_mixed_carryforward_and_discarded_is_kept():
    # origins {A: ancestor-carryforward (not in source), B: discarded (in source, not kept)} -> KEEP on A.
    # The buggy `not any(o in source)` form would drop this; `any(o not in source)` keeps it.
    w = _win(_P, 1, 50, origins=["A", "B"], window_id="wfw-1", line_count=50)
    kept = _filter([w], kept=set(), source={"B"})
    assert kept == [w]


def test_filter_all_discarded_origins_is_dropped():
    w = _win(_P, 1, 50, origins=["B", "C"], window_id="wfw-1", line_count=50)
    assert _filter([w], kept=set(), source={"B", "C"}) == []


def test_filter_kept_origin_is_kept():
    w = _win(_P, 1, 50, origins=["A", "B"], window_id="wfw-1", line_count=50)
    assert _filter([w], kept={"A"}, source={"A", "B"}) == [w]


def test_filter_inactive_path_is_dropped():
    w = _win("/p/other.txt", 1, 50, origins=["A"], window_id="wfw-1", line_count=50)
    assert _filter([w], kept={"A"}, source=set()) == []


# --------------------------------------------------------------------------------------------------------------
# multi-chunk consolidation split (before + after regions)
# --------------------------------------------------------------------------------------------------------------


def test_consolidate_multi_secondary_split():
    result = consolidate_intervals([Interval(1, 500)], Interval(250, 300), 100)
    assert result.primary == Interval(250, 349)  # anchored on new, capped at max_size
    assert [(i.start, i.end) for i in result.secondaries] == [
        (1, 100),
        (101, 200),
        (201, 249),  # before region
        (350, 449),
        (450, 500),  # after region
    ]
    assert result.retired == [Interval(1, 500)]
    # primary + secondaries partition the merged span without overlap
    covered = sorted([result.primary, *result.secondaries], key=lambda i: i.start)
    assert covered[0].start == 1 and covered[-1].end == 500
    for a, b in zip(covered, covered[1:], strict=False):
        assert b.start == a.end + 1


def test_consolidate_chained_exceeds_max():
    result = consolidate_intervals([Interval(1, 80), Interval(81, 160)], Interval(70, 90), 100)
    assert result.primary == Interval(70, 160)
    assert result.secondaries == [Interval(1, 69)]
    assert {(i.start, i.end) for i in result.retired} == {(1, 80), (81, 160)}


# --------------------------------------------------------------------------------------------------------------
# RANGE_TRUNCATED is a response shape only (window source_kind stays read_lines)
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_range_truncated_response_shape(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 400)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    res = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=300), call_id="c-1")
    assert (res.output or "").startswith("RANGE_TRUNCATED")
    assert "start_line=201" in (res.output or "")
    assert len(windows) == 1
    assert windows[0].source_kind == "read_lines"  # NOT range_truncated (dropped from the enum)
    assert (windows[0].view_start_line, windows[0].view_end_line) == (1, 200)


@pytest.mark.asyncio
async def test_overspec_end_on_short_file_returns_file(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 50)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    res = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=300), call_id="c-1")
    assert (res.output or "").startswith("FILE ")
    assert "RANGE_TRUNCATED" not in (res.output or "")
    assert (windows[0].view_start_line, windows[0].view_end_line) == (1, 50)


# --------------------------------------------------------------------------------------------------------------
# write / edit paths set the new required fields
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conflict_diagnostic_has_new_fields(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 10)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    args = _arguments(
        "replace_lines", str(target), expected_revision="deadbeef", start_line=1, end_line=2, new_text="x\n"
    )
    res = await tool.call(args, call_id="w-1")
    assert (res.output or "").startswith("CONFLICT")
    assert len(windows) == 1
    w = windows[0]
    assert w.source_kind == "conflict"
    assert w.origin_call_ids == ["w-1"]
    assert w.line_count == 10
    assert isinstance(w.requested_end_line, int)


@pytest.mark.asyncio
async def test_range_error_diagnostic_has_new_fields(tmp_path: Path):
    # The third diagnostic source_kind (alongside conflict / already_exists): a write whose range is out of bounds
    # but whose revision matches mints a RANGE_ERROR diagnostic via the same `_build_view_carrying_item`. Pin that
    # it carries the new required fields too.
    target = tmp_path / "a.txt"
    _write_lines(target, 10)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="r-1")
    rev = windows[0].revision
    res = await tool.call(
        _arguments("delete_lines", str(target), expected_revision=rev, start_line=50, end_line=60),
        call_id="w-1",
    )
    assert (res.output or "").startswith("RANGE_ERROR")
    diag = next(w for w in windows if w.window_id == "w-1")
    assert diag.source_kind == "range_error"
    assert diag.origin_call_ids == ["w-1"]
    assert diag.line_count == 10
    assert isinstance(diag.requested_end_line, int)


@pytest.mark.asyncio
async def test_successful_edit_refreshes_window_line_count(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 10)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="r-1")
    revision = windows[0].revision
    await tool.call(
        _arguments("delete_lines", str(target), expected_revision=revision, start_line=1, end_line=5),
        call_id="w-1",
    )
    read_window = next(w for w in windows if w.window_id == "r-1")
    assert read_window.line_count == 5  # refreshed to the post-edit count
    assert read_window.view_end_line == 5


# --------------------------------------------------------------------------------------------------------------
# reconcile_working_files end-to-end (fold + obsolete placeholder + tombstone)
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_folds_overlap(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 100)
    windows = [
        _win(str(target), 1, 100, origins=["a"], window_id="a", line_count=100),
        _win(str(target), 50, 60, origins=["b"], window_id="b", line_count=100),
    ]
    await live_windows.reconcile_working_files(windows, tmp_path, max_file_bytes=1_000_000, max_lines=200)
    assert len(windows) == 1
    assert (windows[0].view_start_line, windows[0].view_end_line) == (1, 100)
    assert windows[0].origin_call_ids == ["a", "b"]


@pytest.mark.asyncio
async def test_reconcile_retires_obsolete_placeholder(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 100)  # file now has 100 lines
    placeholder = _win(str(target), 50, 49, origins=["c-1"], window_id="c-1", line_count=10)  # past-EOF when 10 lines
    windows = [placeholder]
    await live_windows.reconcile_working_files(windows, tmp_path, max_file_bytes=1_000_000, max_lines=200)
    assert windows == []  # view_start_line 50 <= new line_count 100 -> no longer past EOF -> retired


@pytest.mark.asyncio
async def test_reconcile_tombstones_missing_file(tmp_path: Path):
    target = tmp_path / "gone.txt"  # never created
    windows = [_win(str(target), 1, 10, origins=["a"], window_id="a", line_count=10)]
    await live_windows.reconcile_working_files(windows, tmp_path, max_file_bytes=1_000_000, max_lines=200)
    assert len(windows) == 1
    assert windows[0].status == "stale"
    assert windows[0].source_kind == "tombstone"


@pytest.mark.asyncio
async def test_create_file_retires_stale_tombstone(tmp_path: Path):
    # read -> delete -> read (tombstone) -> create_file: the stale tombstone must not survive CREATED.
    target = tmp_path / "a.txt"
    _write_lines(target, 3)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=3), call_id="r-1")
    target.unlink()
    err = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=3), call_id="r-2")
    assert (err.output or "").startswith("ERROR")
    assert any(w.source_kind == "tombstone" for w in windows)  # path is tombstoned
    created = await tool.call(_arguments("create_file", str(target), new_text="new\n"), call_id="c-1")
    assert (created.output or "").startswith("CREATED")
    assert not any(w.source_kind == "tombstone" for w in windows)  # tombstone retired


@pytest.mark.asyncio
async def test_write_conflict_retires_stale_tombstone(tmp_path: Path):
    # The recovered-path bug, write variant: a tombstoned path is recovered on disk, then a write with a stale
    # revision hits CONFLICT. `_do_write` must retire the stale tombstone ITSELF — `refresh_windows_for_path`
    # skips stale windows, so without the explicit retire the tombstone would survive alongside the live CONFLICT
    # window and the next <working_files> block would both show the file and claim it inaccessible. No recovering
    # read runs first here, so this is the only test that exercises `_do_write`'s own tombstone retirement
    # (test_write_retires_stale_tombstone_on_recovered_path clears it via a read before the write).
    target = tmp_path / "a.txt"
    _write_lines(target, 10)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="r-1")
    target.unlink()
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="r-2")
    assert any(w.source_kind == "tombstone" for w in windows)
    _write_lines(target, 8)  # path recovers on disk with different content (stale revision -> CONFLICT)
    args = _arguments(
        "replace_lines", str(target), expected_revision="deadbeef", start_line=1, end_line=2, new_text="x\n"
    )
    conflict = await tool.call(args, call_id="w-1")
    assert (conflict.output or "").startswith("CONFLICT")
    assert not any(w.source_kind == "tombstone" for w in windows)  # retired by _do_write, not by a read
    assert any(w.source_kind == "conflict" and w.status == "live" for w in windows)


@pytest.mark.asyncio
async def test_write_retires_stale_tombstone_on_recovered_path(tmp_path: Path):
    # read -> delete -> read (tombstone) -> recreate on disk -> read again: the read clears the tombstone and
    # mints a fresh live window; a follow-up successful edit leaves no tombstone behind.
    target = tmp_path / "a.txt"
    _write_lines(target, 5)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=5), call_id="r-1")
    target.unlink()
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=5), call_id="r-2")
    assert any(w.source_kind == "tombstone" for w in windows)
    _write_lines(target, 5)  # path recovers on disk
    read = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=5), call_id="r-3")
    revision = next(w for w in windows if w.window_id == "r-3").revision
    edit = await tool.call(
        _arguments("delete_lines", str(target), expected_revision=revision, start_line=1, end_line=2),
        call_id="w-1",
    )
    assert (edit.output or "").startswith("EDITED")
    assert not any(w.source_kind == "tombstone" for w in windows)
    assert (read.output or "").startswith("FILE ")


def test_fold_splits_when_merged_exceeds_max_lines():
    text = "\n".join(f"line-{i}" for i in range(1, 251)) + "\n"
    path = "/p/a.txt"
    windows = [
        _win(path, 1, 80, origins=["a"], window_id="a", line_count=250),
        _win(path, 60, 250, origins=["b"], window_id="b", line_count=250),  # overlaps a -> union (1, 250)
    ]
    live_windows.fold_windows_for_path(windows, path, text, "rev", 100)
    assert len(windows) == 3
    assert sorted((w.view_start_line, w.view_end_line) for w in windows) == [(1, 100), (101, 200), (201, 250)]
    assert all(w.origin_call_ids == ["a", "b"] for w in windows)


# --------------------------------------------------------------------------------------------------------------
# TurnExecution builders for the origin-filter call sites (mirror tests/unit_tests/test_origin_filter.py)
# --------------------------------------------------------------------------------------------------------------


def _ft_call(call_id: str) -> TurnItem:
    return TurnItem(type="function_call", call_id=call_id, name="file_tool", arguments="{}")


def _ft_output(call_id: str, path: str) -> TurnItem:
    return TurnItem(
        type="function_call_output",
        call_id=call_id,
        output="...",
        prokaryotes_annotations={"file_tool.persistence": "working_file", "file_tool.path": path},
    )


def _turn(*items: TurnItem) -> dict[str, TurnExecution]:
    return {"b1": TurnExecution(conversation_uuid="c-1", bot_message_source_id="b1", items=list(items))}


# --------------------------------------------------------------------------------------------------------------
# Compaction pre-tail carry-forward filter, generalized to origin_call_ids.
#
# `_cas_swap_child` now delegates to the importable `_carry_forward_windows`, so these tests exercise the REAL
# production predicate (no mirror) over the real `_file_tool_call_ids_in` extraction. They guard the riskiest
# production change: a window survives compaction iff at least one origin escapes the pre-tail span.
# --------------------------------------------------------------------------------------------------------------


def _carried_after_compaction(
    windows: list[WorkingFileWindow], pre_tail_turns: dict[str, TurnExecution]
) -> list[WorkingFileWindow]:
    """Run the production carry-forward filter (`_carry_forward_windows`) the way `_cas_swap_child` does: extract
    pre-tail call_ids via the real `_file_tool_call_ids_in`, then filter."""
    pre_tail_call_ids = _compaction_call_ids_in(pre_tail_turns)
    return _carry_forward_windows(windows, pre_tail_call_ids)


def test_compaction_all_origins_pre_tail_drops():
    w = _win(_P, 1, 50, origins=["A", "B"], window_id="A", line_count=50)
    pre_tail = _turn(_ft_call("A"), _ft_call("B"))
    assert _carried_after_compaction([w], pre_tail) == []


def test_compaction_mixed_pre_tail_and_recency_survives():
    # A is pre-tail (compacted away); B is a recency-tail call absent from pre_tail_turns -> the window rides
    # forward on B. set({A, B}) - {A} = {B} is non-empty.
    w = _win(_P, 1, 50, origins=["A", "B"], window_id="A", line_count=50)
    pre_tail = _turn(_ft_call("A"))
    assert _carried_after_compaction([w], pre_tail) == [w]


@pytest.mark.asyncio
async def test_compaction_redundant_read_anchors_pre_tail_window(tmp_path: Path):
    # End-to-end: a pre-tail read mints the window; a recency-tail REDUNDANT_READ appends its call to the
    # window's origins (real FileTool provenance), so the window survives compaction even though its only
    # minting call is pre-tail. Without the append it would drop.
    target = tmp_path / "a.txt"
    _write_lines(target, 10)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="A")  # pre-tail
    second = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="B")  # recency
    assert (second.output or "").startswith("REDUNDANT_READ")
    w = windows[0]
    assert w.origin_call_ids == ["A", "B"]
    pre_tail = _turn(_ft_call("A"), _ft_output("A", str(target)))  # only A is compacted away
    assert _carried_after_compaction(windows, pre_tail) == [w]  # B (recency) anchors it
    without_append = w.model_copy(update={"origin_call_ids": ["A"]})
    assert _carried_after_compaction([without_append], pre_tail) == []  # A alone -> dropped


@pytest.mark.asyncio
async def test_compaction_diagnostic_mint_all_pre_tail_retires(tmp_path: Path):
    # A write CONFLICT in the pre-tail span mints a diagnostic window with origins == [write_call_id]; on
    # compaction (its only origin pre-tail) it retires.
    target = tmp_path / "a.txt"
    _write_lines(target, 10)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    args = _arguments(
        "replace_lines", str(target), expected_revision="deadbeef", start_line=1, end_line=2, new_text="x\n"
    )
    conflict = await tool.call(args, call_id="w-1")
    assert (conflict.output or "").startswith("CONFLICT")
    diag = windows[0]
    assert diag.origin_call_ids == ["w-1"]
    pre_tail = _turn(_ft_call("w-1"), _ft_output("w-1", str(target)))
    assert _carried_after_compaction(windows, pre_tail) == []
    assert _carried_after_compaction(windows, _turn()) == [diag]  # not pre-tail -> survives


# --------------------------------------------------------------------------------------------------------------
# Branch divergence / cold rebuild origin filter — exercised through the call-site input derivation, not just
# the helper. Reproduces what `_apply_divergence` (conversation_sync.py:337-349) and `_rebuild_from_chain`
# (:603-610) do around `_filter_windows_by_active_path_and_origin`: active paths + kept set from the surviving
# turns, source set from the source/donor turns, all via the real extraction helpers over real TurnExecutions.
# (The inline call-site bodies can't be imported; this pins the wiring they share.)
# --------------------------------------------------------------------------------------------------------------


def _carried_at_call_site(
    windows: list[WorkingFileWindow],
    *,
    kept_turns: dict[str, TurnExecution],
    source_turns: dict[str, TurnExecution],
) -> list[WorkingFileWindow]:
    return _filter_windows_by_active_path_and_origin(
        windows,
        active_paths=_active_paths_in_turns(kept_turns),
        kept_call_ids=_file_tool_call_ids_in(kept_turns),
        source_call_ids=_file_tool_call_ids_in(source_turns),
    )


def test_branch_divergence_ancestor_plus_discarded_merged_window_survives():
    # Merged window origins {A: ancestor-carryforward (in no source turn), B: discarded (in source, not kept)}.
    # The kept branch still touches the path (call K) -> active. Survives on A. The buggy `not any(o in source)`
    # form would drop it; `any(o not in source)` keeps it.
    w = _win(_P, 1, 50, origins=["A", "B"], window_id="wfw-1", line_count=50)
    kept = _turn(_ft_call("K"), _ft_output("K", _P))
    source = _turn(_ft_call("B"), _ft_output("B", _P))
    assert _carried_at_call_site([w], kept_turns=kept, source_turns=source) == [w]


def test_branch_divergence_kept_plus_discarded_merged_window_survives():
    # Over-retention residual: origins {A: kept, B: discarded} -> kept whole on an active path.
    w = _win(_P, 1, 50, origins=["A", "B"], window_id="wfw-1", line_count=50)
    kept = _turn(_ft_call("A"), _ft_output("A", _P))
    source = _turn(_ft_call("A"), _ft_call("B"), _ft_output("A", _P))
    assert _carried_at_call_site([w], kept_turns=kept, source_turns=source) == [w]


def test_branch_divergence_all_discarded_origins_drops():
    w = _win(_P, 1, 50, origins=["B", "C"], window_id="wfw-1", line_count=50)
    kept = _turn(_ft_call("K"), _ft_output("K", _P))  # path active, but no surviving origin
    source = _turn(_ft_call("B"), _ft_call("C"), _ft_output("B", _P))
    assert _carried_at_call_site([w], kept_turns=kept, source_turns=source) == []


def test_branch_divergence_surviving_origin_but_inactive_path_drops():
    # Origin survives but the kept branch no longer touches the path -> the active-path gate drops it.
    w = _win(_P, 1, 50, origins=["A", "B"], window_id="wfw-1", line_count=50)
    kept = _turn(_ft_call("K"), _ft_output("K", "/p/other.txt"))  # active = {/p/other.txt}, not _P
    source = _turn(_ft_call("B"), _ft_output("B", _P))
    assert _carried_at_call_site([w], kept_turns=kept, source_turns=source) == []


def test_cold_rebuild_ancestor_plus_discarded_survives():
    # Same predicate exercised through the cold-rebuild wiring: kept = target turns, source = donor turns.
    w = _win(_P, 1, 50, origins=["A", "B"], window_id="wfw-1", line_count=50)
    target = _turn(_ft_call("T"), _ft_output("T", _P))
    donor = _turn(_ft_call("B"), _ft_output("B", _P))
    assert _carried_at_call_site([w], kept_turns=target, source_turns=donor) == [w]


def _compacted_ancestor_doc(boundary: list, *, snapshot_uuid: str, summary: str) -> dict:
    """A FakeSearchClient-compatible compacted ancestor doc whose boundary_hash matches `boundary`."""
    return {
        "snapshot_uuid": snapshot_uuid,
        "conversation_uuid": "c-1",
        "parent_snapshot_uuid": None,
        "bot_author_id": BOT_ID,
        "compaction_state": "committed",
        "is_compacted": True,
        "summary": summary,
        "ancestor_summaries": [],
        "boundary_hash": compute_boundary_hash(boundary),
        "boundary_message_count": len(boundary),
        "tail_hash": compute_tail_hash(boundary, BOT_ID),
        "messages_json": json.dumps({"messages": [m.model_dump() for m in boundary]}),
        "working_file_windows_json": json.dumps({"windows": []}),
        "raw_message_start_index": 0,
    }


@pytest.mark.asyncio
async def test_rebuild_from_chain_carries_mixed_origin_window_end_to_end():
    # End-to-end cold rebuild through the REAL _rebuild_from_chain (conversation_sync.py:606), not the predicate
    # wrapper: a donor snapshot's windows are hydrated from its doc JSON (exercising WorkingFileWindow.model_validate
    # of the new origin_call_ids field), the call-site derives active paths / kept / source call_ids from real
    # TurnExecutions, and the origin filter runs. Guards wiring + hydration the predicate tests can't reach.
    from prokaryotes.context_v1.conversation_sync import _PartialMessage

    syncer, _redis, search = make_syncer()
    boundary = [msg("1", "u1"), bot_msg("2", "b1")]
    search.conversations["p1"] = _compacted_ancestor_doc(boundary, snapshot_uuid="p1", summary="S1")

    # Donor (latest active child of p1) carries two windows for an active path P: one mixed-origin {A (ancestor-
    # carryforward, ∉ source), B (discarded, ∈ source)} which must SURVIVE on A, and one all-discarded {B} which
    # must DROP — proving the filter actually discriminates, not just passes everything through.
    survivor = _win(_P, 1, 50, origins=["A", "B"], window_id="wfw-survivor", line_count=50)
    dropped = _win(_P, 60, 80, origins=["B"], window_id="wfw-dropped", line_count=80)
    search.conversations["donor"] = {
        "snapshot_uuid": "donor",
        "conversation_uuid": "c-1",
        "parent_snapshot_uuid": "p1",
        "bot_author_id": BOT_ID,
        "compaction_state": "committed",
        "is_compacted": False,
        "summary": None,
        "ancestor_summaries": [],
        "messages_json": json.dumps({"messages": [bot_msg("d-bot", "db").model_dump()]}),
        "working_file_windows_json": json.dumps({"windows": [survivor.model_dump(), dropped.model_dump()]}),
        "raw_message_start_index": 0,
        "dt_modified": "2026-01-01T00:00:00+00:00",
    }
    # Donor turn: file-tool call "B" on path P -> B is a discarded source origin.
    await search.put_turn_execution(
        TurnExecution(
            conversation_uuid="c-1",
            bot_message_source_id="d-bot",
            items=[_ft_call("B"), _ft_output("B", _P)],
        )
    )
    # Target (rebuilt branch) turn: a recency bot turn past the boundary whose file-tool call "T" touches P, so P
    # is active and T is a kept call.
    await search.put_turn_execution(
        TurnExecution(
            conversation_uuid="c-1",
            bot_message_source_id="t-bot",
            items=[_ft_call("T"), _ft_output("T", _P)],
        )
    )

    incoming = [*boundary, bot_msg("t-bot", "recent bot turn")]
    partial = [
        _PartialMessage(
            author_id=m.author_id, content=m.content, client_index=i, source_id=m.source_id, display_name=None
        )
        for i, m in enumerate(incoming)
    ]
    rebuilt = await syncer._rebuild_from_chain(
        conversation_uuid="c-1",
        snapshot_uuid="p1",
        bot_author_id=BOT_ID,
        partial=partial,
        head_doc=None,
    )

    assert rebuilt.parent_snapshot_uuid == "p1"
    carried = rebuilt.working_file_windows
    assert [w.window_id for w in carried] == ["wfw-survivor"]  # mixed-origin survives, all-discarded dropped
    assert carried[0].origin_call_ids == ["A", "B"]  # hydrated through the doc round-trip


# --------------------------------------------------------------------------------------------------------------
# Read-flow coverage gaps: EOF-at-cap boundary, multi-window subset, diagnostic coverage scoping, non-healing.
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_eof_at_cap_boundary_prefers_eof_message(tmp_path: Path):
    # 200-line file with max_lines=200: line_count == cap_end_line. A re-read (1, 300) is covered, and the EOF
    # branch must win over the cap branch (file simply ends; paging would loop past EOF).
    target = tmp_path / "a.txt"
    _write_lines(target, 200)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=200), call_id="c-1")
    second = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=300), call_id="c-2")
    out = second.output or ""
    assert out.startswith("REDUNDANT_READ"), out
    assert "File ends at line 200" in out, out
    assert "page forward from start_line=" not in out, out  # the cap-style guidance must NOT appear


@pytest.mark.asyncio
async def test_multi_window_subset_consolidates_not_redundant(tmp_path: Path):
    # Two non-mergeable contiguous windows (page split) whose union covers a request that no single window does.
    # Must consolidate (FILE), not short-circuit REDUNDANT (v1 only checks single-window coverage).
    target = tmp_path / "a.txt"
    _write_lines(target, 400)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=200), call_id="c-1")
    await tool.call(_arguments("read_lines", str(target), start_line=201, end_line=400), call_id="c-2")
    assert len(windows) == 2  # contiguous-but-separate (don't fit in one)
    third = await tool.call(_arguments("read_lines", str(target), start_line=150, end_line=300), call_id="c-3")
    out = third.output or ""
    assert out.startswith("FILE "), out
    assert "REDUNDANT" not in out, out
    # primary owns the requested span (anchored on new); the union is covered without overlap
    by_start = {w.view_start_line: w for w in windows}
    assert 150 in by_start and by_start[150].window_id == "c-3"
    covered = sorted((w.view_start_line, w.view_end_line) for w in windows)
    for a, b in zip(covered, covered[1:], strict=False):
        assert b[0] == a[1] + 1  # contiguous, non-overlapping cover


@pytest.mark.asyncio
async def test_refreshed_unreached_diagnostic_grants_later_coverage(tmp_path: Path):
    # Scoping property: the pre-refresh diagnostic exclusion is per-read, NOT absolute. A diagnostic that a
    # disjoint read normalizes-but-doesn't-reach becomes an ordinary read_lines window and may grant REDUNDANT
    # to a *later* same-turn read.
    target = tmp_path / "a.txt"
    _write_lines(target, 300)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    # CONFLICT at line 1 mints a diagnostic covering (1, 200) (capped at max_lines), no prior read window.
    conflict = await tool.call(
        _arguments(
            "replace_lines", str(target), expected_revision="deadbeef", start_line=1, end_line=2, new_text="x\n"
        ),
        call_id="w-1",
    )
    assert (conflict.output or "").startswith("CONFLICT")
    assert windows[0].source_kind == "conflict"
    # Disjoint read past the diagnostic's range -> normalizes it to read_lines but does not reach it.
    disjoint = await tool.call(_arguments("read_lines", str(target), start_line=250, end_line=260), call_id="r-1")
    assert (disjoint.output or "").startswith("FILE ")
    assert next(w for w in windows if w.window_id == "w-1").source_kind == "read_lines"  # normalized
    # A later read inside the now-read_lines window -> REDUNDANT (coverage granted to a later read).
    later = await tool.call(_arguments("read_lines", str(target), start_line=5, end_line=10), call_id="r-2")
    assert (later.output or "").startswith("REDUNDANT_READ"), later.output or ""


@pytest.mark.asyncio
async def test_coverage_hit_does_not_heal_same_turn_overlap(tmp_path: Path):
    # Accepted non-healing residual: a write-minted diagnostic transiently overlaps a read_lines window; a later
    # covered read returns REDUNDANT via the read_lines window and leaves the overlap intact. The next reconcile
    # fold collapses it. Guards against anyone adding read-path healing for the coverage short-circuit.
    target = tmp_path / "a.txt"
    _write_lines(target, 100)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=100), call_id="c-1")  # (1,100)
    # CONFLICT at line 50 mints a diagnostic (50,100), overlapping the (1,100) read window.
    conflict = await tool.call(
        _arguments(
            "replace_lines", str(target), expected_revision="deadbeef", start_line=50, end_line=51, new_text="x\n"
        ),
        call_id="w-2",
    )
    assert (conflict.output or "").startswith("CONFLICT")
    assert len(windows) == 2
    assert sorted((w.view_start_line, w.view_end_line) for w in windows) == [(1, 100), (50, 100)]  # overlap
    later = await tool.call(_arguments("read_lines", str(target), start_line=55, end_line=56), call_id="c-3")
    assert (later.output or "").startswith("REDUNDANT_READ")  # covered via the (1,100) read_lines window
    assert len(windows) == 2  # overlap persists — the coverage short-circuit healed nothing
    assert sorted((w.view_start_line, w.view_end_line) for w in windows) == [(1, 100), (50, 100)]
    # the next reconcile fold collapses it
    await live_windows.reconcile_working_files(windows, tmp_path, max_file_bytes=1_000_000, max_lines=200)
    assert len(windows) == 1
    assert (windows[0].view_start_line, windows[0].view_end_line) == (1, 100)


@pytest.mark.asyncio
async def test_past_eof_diagnostic_then_same_range_read_is_benign(tmp_path: Path):
    # A diagnostic whose view_start_line is past EOF is itself an empty-view window; it is unreachable by
    # consolidation and excluded by the reconcile fold. A same-range past-EOF read normalizes it to read_lines
    # (still empty-view) and mints its own placeholder — two empty-views coexist. Pin this as the documented
    # benign residual: empty-views cover no real line (no non-overlap violation), and reconcile retires both
    # once the file grows past their start.
    target = tmp_path / "a.txt"
    _write_lines(target, 10)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    # CONFLICT at a past-EOF start (no prior read) -> empty-view diagnostic; nothing covers EOF.
    conflict = await tool.call(
        _arguments(
            "replace_lines", str(target), expected_revision="deadbeef", start_line=50, end_line=51, new_text="x\n"
        ),
        call_id="w-1",
    )
    assert (conflict.output or "").startswith("CONFLICT")
    diag = windows[0]
    assert diag.source_kind == "conflict" and diag.view_end_line < diag.view_start_line
    read = await tool.call(_arguments("read_lines", str(target), start_line=50, end_line=60), call_id="r-2")
    assert "line_count=10" in (read.output or "") and "REDUNDANT" not in (read.output or "")
    assert next(w for w in windows if w.window_id == "w-1").source_kind == "read_lines"  # normalized, not folded
    content = [(w.view_start_line, w.view_end_line) for w in windows if w.view_end_line >= w.view_start_line]
    empties = [w for w in windows if w.view_end_line < w.view_start_line]
    assert content == []  # no content window double-covers any line
    assert len(empties) == 2  # benign coexistence: normalized diagnostic + new placeholder
    _write_lines(target, 80)  # file grows past line 50
    await live_windows.reconcile_working_files(windows, tmp_path, max_file_bytes=1_000_000, max_lines=200)
    assert windows == []  # both obsolete empty-views retired


# --------------------------------------------------------------------------------------------------------------
# Regression: a content window that transiently shrinks below its start line (file shrinks) must REGROW on a
# later file growth, not be retired as an obsolete past-EOF placeholder. The retirement check must key on
# post-render geometry, not the pre-render is_empty_view flag (live_windows.refresh_windows_for_path).
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_regrown_content_window_survives(tmp_path: Path):
    # window (51,80) -> external shrink to 40 lines (window goes empty-view (51,50)) -> external regrow to 100
    # lines. Lines 51-80 have content again, so the window must resync to (51,80), NOT be dropped.
    target = tmp_path / "a.txt"
    _write_lines(target, 80)
    windows = [_win(str(target), 51, 80, origins=["c-1"], window_id="c-1", line_count=80)]
    _write_lines(target, 40)  # turn N: external shrink below the window's start
    await live_windows.reconcile_working_files(windows, tmp_path, max_file_bytes=1_000_000, max_lines=200)
    assert [(w.view_start_line, w.view_end_line) for w in windows] == [(51, 50)]  # transiently empty-view
    _write_lines(target, 100)  # turn N+1: external regrow above the window's start
    await live_windows.reconcile_working_files(windows, tmp_path, max_file_bytes=1_000_000, max_lines=200)
    assert len(windows) == 1  # NOT retired
    w = windows[0]
    assert (w.view_start_line, w.view_end_line) == (51, 80)  # resynced to content
    assert w.line_count == 100
    assert w.window_id == "c-1"  # identity preserved (fold pass-through singleton)


@pytest.mark.asyncio
async def test_insert_regrow_keeps_shrunk_window(tmp_path: Path):
    # Write-path variant through `_do_write`: a read window shrinks to empty-view via a delete, then an insert
    # regrows the file past its start. `_do_write`'s `refresh_windows_for_path` must resync it, not drop it.
    target = tmp_path / "a.txt"
    _write_lines(target, 80)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=51, end_line=80), call_id="c-1")
    rev80 = next(w for w in windows if w.window_id == "c-1").revision
    # delete lines 1-40 -> file shrinks to 40; the (51,80) window becomes empty-view (51,50)
    await tool.call(
        _arguments("delete_lines", str(target), expected_revision=rev80, start_line=1, end_line=40),
        call_id="w-1",
    )
    c1 = next(w for w in windows if w.window_id == "c-1")
    assert (c1.view_start_line, c1.view_end_line) == (51, 50)  # shrunk to empty-view
    rev40 = c1.revision
    # append 60 lines at EOF -> file regrows to 100; lines 51-80 have content again
    new_block = "\n".join(f"add-{i}" for i in range(1, 61)) + "\n"
    await tool.call(
        _arguments("insert_lines", str(target), expected_revision=rev40, start_line=41, new_text=new_block),
        call_id="w-2",
    )
    survivors = [w for w in windows if w.window_id == "c-1"]
    assert len(survivors) == 1  # NOT dropped by the insert's refresh
    assert (survivors[0].view_start_line, survivors[0].view_end_line) == (51, 80)
    assert survivors[0].line_count == 100


# --------------------------------------------------------------------------------------------------------------
# Constructor-seeded exposure (the cross-turn dedup path). `FileTool.__init__` marks every window the provider
# returns as exposed at its current revision because they are about to be projected. A covered read on a fresh
# instance must REDUNDANT via that seed — distinct from the per-read `_mark_exposed` every other coverage test
# exercises. The negative pins that the seed is captured at construction and goes stale on a later file change.
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_constructor_seeded_window_grants_redundant_cross_turn(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 10)
    windows: list[WorkingFileWindow] = []
    tool1 = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool1.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="t1-c1")
    assert len(windows) == 1
    # Next turn: a fresh FileTool is constructed over the carried windows. tool2 has never rendered anything, so a
    # REDUNDANT here can only come from the __init__ seed marking the carried window exposed at its revision.
    tool2 = FileTool(lambda: windows, workspace_root=tmp_path)
    second = await tool2.call(_arguments("read_lines", str(target), start_line=2, end_line=6), call_id="t2-c1")
    assert (second.output or "").startswith("REDUNDANT_READ")
    assert len(windows) == 1
    assert windows[0].origin_call_ids == ["t1-c1", "t2-c1"]


@pytest.mark.asyncio
async def test_constructor_seed_goes_stale_when_file_changes_before_read(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 10)
    windows: list[WorkingFileWindow] = []
    tool1 = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool1.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="t1-c1")
    # Fresh instance seeds exposure at the carried revision...
    tool2 = FileTool(lambda: windows, workspace_root=tmp_path)
    target.write_text("\n".join(f"X-{i}" for i in range(1, 11)) + "\n", encoding="utf-8")  # same 10 lines, new content
    # ...but the file changed before the read. The per-read refresh advances the window's revision, so the seed is
    # now stale and the covered range must render current content, not REDUNDANT.
    second = await tool2.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="t2-c1")
    assert (second.output or "").startswith("FILE ")
    assert "REDUNDANT" not in (second.output or "")
    assert "X-1" in (second.output or "")


# --------------------------------------------------------------------------------------------------------------
# Write-refresh exposure: `_do_write`'s `refresh_windows_for_path` advances an existing read window's revision and
# re-renders it WITHOUT marking it exposed. A same-turn covered re-read after a successful edit must surface the
# post-edit content rather than a REDUNDANT pointer to a window the model has not been re-shown this turn.
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reread_after_edit_renders_current_not_redundant(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 10)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="r-1")
    rev = windows[0].revision
    edit = await tool.call(
        _arguments("replace_lines", str(target), expected_revision=rev, start_line=1, end_line=1, new_text="CHANGED\n"),
        call_id="w-1",
    )
    assert (edit.output or "").startswith("EDITED")
    reread = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="r-2")
    assert (reread.output or "").startswith("FILE ")
    assert "REDUNDANT" not in (reread.output or "")
    assert "CHANGED" in (reread.output or "")  # post-edit content surfaced


# --------------------------------------------------------------------------------------------------------------
# Empty-FILE placeholder `(1, 0)` retirement via reconcile. Distinct render path from the past-EOF `(start,
# start - 1)` shape: `requested_end_line == 0`, so the empty view comes from `min(line_count, 0)` rather than from
# a past-EOF start. Once the file gains content, reconcile must retire it.
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_retires_empty_file_placeholder_when_content_added(tmp_path: Path):
    target = tmp_path / "a.txt"
    target.write_text("", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    first = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="c-1")
    assert "line_count=0" in (first.output or "")
    assert (windows[0].view_start_line, windows[0].view_end_line, windows[0].requested_end_line) == (1, 0, 0)
    _write_lines(target, 5)  # file gains content out of band
    await live_windows.reconcile_working_files(windows, tmp_path, max_file_bytes=1_000_000, max_lines=200)
    assert windows == []  # (1, 0) placeholder retired now that line 1 has content


# --------------------------------------------------------------------------------------------------------------
# RANGE_TRUNCATED consolidating against a pre-existing overlapping window. The other RANGE_TRUNCATED test starts
# from an empty window list; this pins that the over-cap read still folds an existing window into one (1, 200)
# read_lines window with merged origins while emitting the RANGE_TRUNCATED response shape.
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_range_truncated_consolidates_existing_window(tmp_path: Path):
    target = tmp_path / "a.txt"
    _write_lines(target, 400)
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=tmp_path)
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=50), call_id="c-1")
    res = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=300), call_id="c-2")
    assert (res.output or "").startswith("RANGE_TRUNCATED")
    assert "start_line=201" in (res.output or "")
    assert len(windows) == 1
    w = windows[0]
    assert (w.view_start_line, w.view_end_line) == (1, 200)
    assert w.source_kind == "read_lines"
    assert w.window_id == "c-2"
    assert w.origin_call_ids == ["c-1", "c-2"]


# --------------------------------------------------------------------------------------------------------------
# Storage round-trip for the new-shape window. Persistence is whole-model `model_dump()` -> JSON -> `model_validate`
# (search_v1.conversations), so the new required fields survive automatically; the old-shape rejection pins the
# documented no-backcompat stance (old snapshots are dropped, not silently coerced).
# --------------------------------------------------------------------------------------------------------------


def test_working_file_window_storage_round_trip():
    w = _win(_P, 1, 50, origins=["c-2", "c-1"], window_id="c-1", line_count=80)
    payload = json.dumps({"windows": [w.model_dump()]})  # mirrors put_conversation
    restored = [WorkingFileWindow.model_validate(item) for item in json.loads(payload)["windows"]]  # mirrors hydrate
    assert restored == [w]
    assert restored[0].line_count == 80
    assert restored[0].origin_call_ids == ["c-1", "c-2"]
    assert restored[0].requested_end_line == 50


def test_old_shape_window_dict_rejected_on_hydrate():
    old = {
        "window_id": "c-1",
        "path": _P,
        "status": "live",
        "revision": "r",
        "rendered_output": "",
        "view_start_line": 1,
        "view_end_line": 10,
        "requested_end_line": None,  # old optional default — now required int
        "source_kind": "range_truncated",  # dropped from the enum
        # no line_count, no origin_call_ids — both now required
    }
    with pytest.raises(ValidationError):
        WorkingFileWindow.model_validate(old)


# --------------------------------------------------------------------------------------------------------------
# Multi-path `reconcile_working_files`: each distinct path is refreshed/folded/tombstoned independently in the
# `for path_str in sorted(live_paths)` loop. One path folds an overlap, one tombstones (file gone), one regrows.
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_multi_path_independent(tmp_path: Path):
    a = tmp_path / "a.txt"
    _write_lines(a, 100)
    b = tmp_path / "b.txt"  # never created -> tombstone
    c = tmp_path / "c.txt"
    _write_lines(c, 100)
    windows = [
        _win(str(a), 1, 100, origins=["a1"], window_id="a1", line_count=100),
        _win(str(a), 50, 60, origins=["a2"], window_id="a2", line_count=100),  # overlaps a1 -> folds
        _win(str(b), 1, 10, origins=["b1"], window_id="b1", line_count=10),  # file gone -> tombstone
        _win(str(c), 51, 80, origins=["c1"], window_id="c1", line_count=80),  # stale line_count -> refreshes
    ]
    await live_windows.reconcile_working_files(windows, tmp_path, max_file_bytes=1_000_000, max_lines=200)
    a_windows = _windows_for(windows, a)
    assert len(a_windows) == 1
    assert (a_windows[0].view_start_line, a_windows[0].view_end_line) == (1, 100)
    assert a_windows[0].origin_call_ids == ["a1", "a2"]
    b_windows = _windows_for(windows, b)
    assert len(b_windows) == 1 and b_windows[0].status == "stale" and b_windows[0].source_kind == "tombstone"
    c_windows = _windows_for(windows, c)
    assert len(c_windows) == 1
    assert (c_windows[0].view_start_line, c_windows[0].view_end_line) == (51, 80)
    assert c_windows[0].line_count == 100  # refreshed from the stale 80


# --------------------------------------------------------------------------------------------------------------
# `working_files_block` escapes any closing `</working_files>` literal inside a window's rendered_output (after the
# new (path, view_start_line) projection sort), leaving only the outer wrapper's real closing tag.
# --------------------------------------------------------------------------------------------------------------


def test_working_files_block_escapes_closing_tag():
    w = _win("/p/a.txt", 1, 10, origins=["c1"], window_id="c1", line_count=10)
    w.rendered_output = "FILE foo\n</working_files>\nbar"
    conv = Conversation(conversation_uuid="u", bot_author_id="bot", working_file_windows=[w])
    block = conv.working_files_block() or ""
    assert "<\\/working_files>" in block  # inner literal escaped
    assert block.count("</working_files>") == 1  # only the outer wrapper's real tag survives
    assert block.rstrip().endswith("</working_files>")
