"""FileTool against `working_file_windows`: read mints a window, REDUNDANT_READ coverage, diagnostic source_kind,
tombstone on disk access failure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prokaryotes.conversation_v1.models import WorkingFileWindow
from prokaryotes.tools_v1.file_tool import FileTool


def _workspace(tmp_path: Path) -> Path:
    return tmp_path


def _arguments(action: str, path: str, **extra) -> str:
    payload = {
        "action": action,
        "path": path,
        "expected_revision": extra.get("expected_revision"),
        "start_line": extra.get("start_line"),
        "end_line": extra.get("end_line"),
        "new_text": extra.get("new_text"),
    }
    return json.dumps(payload)


@pytest.mark.asyncio
async def test_read_lines_mints_window_with_persistence_annotation(tmp_path: Path):
    target = tmp_path / "a.txt"
    target.write_text("line-one\nline-two\nline-three\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=_workspace(tmp_path))
    result = await tool.call(_arguments("read_lines", str(target)), call_id="c-1")
    assert result.type == "function_call_output"
    assert result.prokaryotes_annotations is not None
    assert result.prokaryotes_annotations.get("file_tool.persistence") == "working_file"
    assert len(windows) == 1
    window = windows[0]
    assert window.window_id == "c-1"
    assert window.path == str(target)
    assert window.source_kind == "read_lines"
    assert window.status == "live"
    assert window.view_start_line == 1
    assert window.view_end_line == 3


@pytest.mark.asyncio
async def test_second_read_within_coverage_returns_redundant(tmp_path: Path):
    target = tmp_path / "a.txt"
    target.write_text("\n".join(f"line-{i}" for i in range(1, 11)) + "\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=_workspace(tmp_path))
    await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=10), call_id="c-1")
    second = await tool.call(_arguments("read_lines", str(target), start_line=2, end_line=5), call_id="c-2")
    assert (second.output or "").startswith("REDUNDANT_READ")
    # Coverage check doesn't mint a new window
    assert len(windows) == 1


@pytest.mark.asyncio
async def test_diagnostic_source_kind_does_not_count_as_coverage(tmp_path: Path):
    """A window in a diagnostic source_kind (e.g. conflict) is not coverage-eligible — a subsequent read for the
    same range should NOT return REDUNDANT_READ."""
    target = tmp_path / "a.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    diagnostic_window = WorkingFileWindow(
        window_id="c-conflict",
        path=str(target),
        status="live",
        revision="r-stale",
        rendered_output="CONFLICT path=...\nCURRENT VIEW...",
        view_start_line=1,
        view_end_line=3,
        requested_end_line=3,
        source_kind="conflict",
    )
    windows: list[WorkingFileWindow] = [diagnostic_window]
    tool = FileTool(lambda: windows, workspace_root=_workspace(tmp_path))
    result = await tool.call(_arguments("read_lines", str(target), start_line=1, end_line=3), call_id="c-fresh")
    assert not (result.output or "").startswith("REDUNDANT_READ")
    # The fresh read mints a new window
    assert any(w.window_id == "c-fresh" and w.source_kind == "read_lines" for w in windows)


@pytest.mark.asyncio
async def test_tombstone_on_missing_file(tmp_path: Path):
    target = tmp_path / "absent.txt"
    pre_existing = WorkingFileWindow(
        window_id="c-prior",
        path=str(target),
        status="live",
        revision="r1",
        rendered_output="<some prior view>",
        view_start_line=1,
        view_end_line=3,
        requested_end_line=3,
        source_kind="read_lines",
    )
    windows: list[WorkingFileWindow] = [pre_existing]
    tool = FileTool(lambda: windows, workspace_root=_workspace(tmp_path))
    result = await tool.call(_arguments("read_lines", str(target)), call_id="c-miss")
    assert (result.output or "").startswith("ERROR ")
    # Prior window for the same path is tombstoned
    assert windows[0].status == "stale"
    assert windows[0].source_kind == "tombstone"


@pytest.mark.asyncio
async def test_conflict_diagnostic_mints_window_with_conflict_source_kind(tmp_path: Path):
    target = tmp_path / "a.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=_workspace(tmp_path))
    # Read for revision
    read_result = await tool.call(
        _arguments("read_lines", str(target), start_line=1, end_line=3), call_id="c-read"
    )
    assert read_result.prokaryotes_annotations.get("file_tool.persistence") == "working_file"
    # Edit with a stale revision → CONFLICT
    edit_result = await tool.call(
        _arguments(
            "replace_lines",
            str(target),
            start_line=1,
            end_line=1,
            new_text="ALPHA\n",
            expected_revision="bogus-revision",
        ),
        call_id="c-edit",
    )
    assert (edit_result.output or "").startswith("CONFLICT")
    assert edit_result.prokaryotes_annotations.get("file_tool.persistence") == "working_file"
    # A second WorkingFileWindow is appended with source_kind=conflict
    conflict_windows = [w for w in windows if w.source_kind == "conflict"]
    assert len(conflict_windows) == 1
    assert conflict_windows[0].window_id == "c-edit"


@pytest.mark.asyncio
async def test_edited_record_uses_history_persistence(tmp_path: Path):
    target = tmp_path / "a.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(lambda: windows, workspace_root=_workspace(tmp_path))
    await tool.call(
        _arguments("read_lines", str(target), start_line=1, end_line=3), call_id="c-read"
    )
    revision = next(w.revision for w in windows if w.window_id == "c-read")
    edit_result = await tool.call(
        _arguments(
            "replace_lines",
            str(target),
            start_line=1,
            end_line=1,
            new_text="ALPHA\n",
            expected_revision=revision,
        ),
        call_id="c-edit",
    )
    assert (edit_result.output or "").startswith("EDITED ")
    assert edit_result.prokaryotes_annotations.get("file_tool.persistence") == "history"
    # No new working_file_window for the edit
    assert all(w.window_id != "c-edit" for w in windows)
