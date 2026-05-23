import asyncio
import fcntl
import json
import threading
from hashlib import sha256
from pathlib import Path

import pytest

from prokaryotes.conversation_v1.models import WorkingFileWindow
from prokaryotes.tools_v1.file_tool import FileTool, reads
from prokaryotes.tools_v1.file_tool.live_windows import (
    reconcile_working_files,
    refresh_windows_for_path,
)
from prokaryotes.tools_v1.file_tool.reads import _locked_read_text
from prokaryotes.tools_v1.file_tool.rendering import render_view


def _hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _read_args(path: Path, start_line: int | None = None, end_line: int | None = None) -> str:
    return json.dumps(
        {
            "action": "read_lines",
            "path": str(path),
            "expected_revision": None,
            "start_line": start_line,
            "end_line": end_line,
            "new_text": None,
        }
    )


def _replace_args(path: Path, expected_revision: str, start: int, end: int, new_text: str) -> str:
    return json.dumps(
        {
            "action": "replace_lines",
            "path": str(path),
            "expected_revision": expected_revision,
            "start_line": start,
            "end_line": end,
            "new_text": new_text,
        }
    )


def _create_args(path: Path, new_text: str) -> str:
    return json.dumps(
        {
            "action": "create_file",
            "path": str(path),
            "expected_revision": None,
            "start_line": None,
            "end_line": None,
            "new_text": new_text,
        }
    )


def _insert_args(path: Path, expected_revision: str, start: int, new_text: str) -> str:
    return json.dumps(
        {
            "action": "insert_lines",
            "path": str(path),
            "expected_revision": expected_revision,
            "start_line": start,
            "end_line": None,
            "new_text": new_text,
        }
    )


def _delete_args(path: Path, expected_revision: str, start: int, end: int) -> str:
    return json.dumps(
        {
            "action": "delete_lines",
            "path": str(path),
            "expected_revision": expected_revision,
            "start_line": start,
            "end_line": end,
            "new_text": None,
        }
    )


@pytest.mark.asyncio
async def test_read_returns_live_window_with_annotations(tmp_path: Path):
    target = tmp_path / "hello.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    result = await tool.call(_read_args(target), "call_1")

    assert result.type == "function_call_output"
    assert result.call_id == "call_1"
    assert result.prokaryotes_annotations["file_tool.path"] == str(target.resolve())
    assert result.prokaryotes_annotations["file_tool.persistence"] == "working_file"
    assert "1 | alpha" in result.output
    assert "line_count=3" in result.output
    # Per-call window state lives on the WorkingFileWindow now, not on the output annotations.
    assert len(windows) == 1
    window = windows[0]
    assert window.window_id == "call_1"
    assert window.path == str(target.resolve())
    assert window.status == "live"
    assert window.source_kind == "read_lines"
    assert window.view_start_line == 1
    assert window.view_end_line == 3
    assert window.revision == _hash("alpha\nbeta\ngamma\n")


@pytest.mark.asyncio
async def test_read_with_exact_end_line_returns_requested_span(tmp_path: Path):
    target = tmp_path / "span.txt"
    target.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    result = await tool.call(_read_args(target, start_line=2, end_line=3), "call_span")

    assert "2 | beta" in result.output
    assert "3 | gamma" in result.output
    assert "4 | delta" not in result.output
    window = windows[-1]
    assert window.view_start_line == 2
    assert window.view_end_line == 3
    assert window.requested_end_line == 3


@pytest.mark.asyncio
async def test_read_empty_file_yields_zero_line_count(tmp_path: Path):
    target = tmp_path / "empty.txt"
    target.write_text("", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    result = await tool.call(_read_args(target), "call_e")

    assert "line_count=0" in result.output
    assert "1 | " not in result.output
    assert windows[-1].status == "live"
    assert windows[-1].source_kind == "read_lines"


@pytest.mark.asyncio
async def test_read_missing_file_returns_error(tmp_path: Path):
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    result = await tool.call(_read_args(tmp_path / "missing.txt"), "call_x")

    assert result.output.startswith("ERROR FileNotFoundError")
    assert result.prokaryotes_annotations is None


@pytest.mark.asyncio
async def test_read_path_escape_returns_error(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope\n", encoding="utf-8")
    tool = FileTool(working_file_provider=lambda: [], workspace_root=workspace)

    result = await tool.call(_read_args(outside), "call_esc")

    assert result.output.startswith("ERROR ValueError")
    assert "escapes workspace root" in result.output


@pytest.mark.asyncio
async def test_read_rejects_non_positive_start_line(tmp_path: Path):
    target = tmp_path / "hello.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    zero_result = await tool.call(_read_args(target, start_line=0), "call_zero")
    negative_result = await tool.call(_read_args(target, start_line=-3), "call_neg")

    assert zero_result.output.startswith("ERROR ValueError")
    assert "start_line for read_lines" in zero_result.output
    assert zero_result.prokaryotes_annotations is None
    assert negative_result.output.startswith("ERROR ValueError")
    assert "start_line for read_lines" in negative_result.output
    assert negative_result.prokaryotes_annotations is None


@pytest.mark.asyncio
async def test_read_rejects_invalid_end_line(tmp_path: Path):
    target = tmp_path / "hello.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    zero_result = await tool.call(_read_args(target, start_line=1, end_line=0), "call_end_zero")
    reversed_result = await tool.call(_read_args(target, start_line=3, end_line=2), "call_end_rev")

    assert zero_result.output.startswith("ERROR ValueError")
    assert "end_line for read_lines" in zero_result.output
    assert reversed_result.output.startswith("ERROR ValueError")
    assert "end_line for read_lines must be >= start_line" in reversed_result.output


@pytest.mark.asyncio
async def test_read_over_cap_with_truncation_returns_range_truncated_live_view(tmp_path: Path):
    target = tmp_path / "long.txt"
    line_count = FileTool.max_lines + 55
    target.write_text("".join(f"line{i}\n" for i in range(1, line_count + 1)), encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    result = await tool.call(
        _read_args(target, start_line=1, end_line=line_count),
        "call_truncated",
    )

    cap_end = FileTool.max_lines
    assert result.output.startswith(
        f"RANGE_TRUNCATED path={target.resolve()} requested_lines=1-{line_count}"
        f" returned_lines=1-{cap_end} line_count={line_count}"
    )
    assert f"Call `read_lines` with `start_line={cap_end + 1}`" in result.output
    assert f"remaining {line_count - cap_end} lines" in result.output
    assert "1 | line1" in result.output
    assert f"{cap_end} | line{cap_end}" in result.output
    assert f"{cap_end + 1} | line{cap_end + 1}" not in result.output
    window = windows[-1]
    assert window.source_kind == "range_truncated"
    assert window.view_start_line == 1
    assert window.view_end_line == cap_end
    assert window.requested_end_line == cap_end


@pytest.mark.asyncio
async def test_read_over_cap_remaining_count_tracks_requested_span_not_eof(tmp_path: Path):
    # File is much larger than the requested span; remaining count should describe the remainder of the *requested*
    # span (requested_end - cap_end), not the rest of the file. Otherwise the model is nudged into paging past what it
    # actually asked for.
    target = tmp_path / "much_longer.txt"
    file_line_count = FileTool.max_lines * 5
    requested_end = FileTool.max_lines + 50
    target.write_text("".join(f"line{i}\n" for i in range(1, file_line_count + 1)), encoding="utf-8")
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    result = await tool.call(
        _read_args(target, start_line=1, end_line=requested_end),
        "call_remaining",
    )

    assert result.output.startswith(
        f"RANGE_TRUNCATED path={target.resolve()} requested_lines=1-{requested_end}"
        f" returned_lines=1-{FileTool.max_lines} line_count={file_line_count}"
    )
    assert f"remaining {requested_end - FileTool.max_lines} lines" in result.output
    assert f"remaining {file_line_count - FileTool.max_lines} lines" not in result.output


@pytest.mark.asyncio
async def test_read_over_cap_with_eof_inside_cap_returns_plain_live_view(tmp_path: Path):
    target = tmp_path / "short.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    result = await tool.call(
        _read_args(target, start_line=1, end_line=FileTool.max_lines + 1),
        "call_over_cap_short",
    )

    # Cap hit at the request layer but EOF (line 3) is within the cap: nothing truncated, so the model gets a plain FILE
    # view — no RANGE_TRUNCATED.
    assert result.output.startswith("FILE ")
    assert "RANGE_TRUNCATED" not in result.output
    assert "1 | alpha" in result.output
    assert "3 | gamma" in result.output
    window = windows[-1]
    assert window.view_end_line == 3
    # `requested_end_line` is pinned to the cap, not the original over-cap request, so a later refresh after file
    # growth cannot expand the window past max_lines.
    assert window.requested_end_line == FileTool.max_lines


@pytest.mark.asyncio
async def test_read_over_cap_pinned_requested_end_caps_refresh_after_file_growth(tmp_path: Path):
    target = tmp_path / "growth.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    await tool.call(
        _read_args(target, start_line=1, end_line=FileTool.max_lines + 100),
        "call_over_cap_grow",
    )

    # Grow the file beyond the cap; the pinned requested_end_line must prevent the refreshed window from
    # extending past max_lines, even though the original request was wider.
    grown_line_count = FileTool.max_lines + 50
    target.write_text("".join(f"L{i}\n" for i in range(1, grown_line_count + 1)), encoding="utf-8")
    await reconcile_working_files(
        windows,
        workspace_root=tmp_path,
        max_file_bytes=FileTool.max_file_bytes,
        max_lines=FileTool.max_lines,
    )

    refreshed = windows[0]
    assert refreshed.view_end_line == FileTool.max_lines
    assert f"{FileTool.max_lines} | L{FileTool.max_lines}" in refreshed.rendered_output
    assert f"{FileTool.max_lines + 1} | L{FileTool.max_lines + 1}" not in refreshed.rendered_output


@pytest.mark.asyncio
async def test_default_workspace_root_is_current_working_directory(tmp_path: Path, monkeypatch):
    target = tmp_path / "relative.txt"
    target.write_text("from cwd\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    tool = FileTool(working_file_provider=lambda: [])

    result = await tool.call(_read_args(Path("relative.txt")), "call_rel")

    assert result.prokaryotes_annotations["file_tool.path"] == str(target.resolve())
    assert "1 | from cwd" in result.output


@pytest.mark.asyncio
async def test_create_file_writes_new_file_and_returns_created_record(tmp_path: Path):
    target = tmp_path / "created.txt"
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    result = await tool.call(_create_args(target, "alpha\nbeta\n"), "call_create")

    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"
    # CREATED records ride the transcript forward as history; they do NOT mint a working_file_window.
    assert result.prokaryotes_annotations == {
        "file_tool.path": str(target.resolve()),
        "file_tool.persistence": "history",
    }
    assert result.output.startswith("CREATED ")
    assert "revision: " in result.output
    assert "line_count: 0 → 2" in result.output
    assert "Added (lines 1-2):" in result.output
    assert windows == []


@pytest.mark.asyncio
async def test_create_file_creates_missing_parent_directories(tmp_path: Path):
    target = tmp_path / "nested" / "deeper" / "created.txt"
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    result = await tool.call(_create_args(target, "alpha\n"), "call_create_nested")

    assert target.read_text(encoding="utf-8") == "alpha\n"
    assert target.parent.is_dir()
    assert result.output.startswith("CREATED ")
    assert result.prokaryotes_annotations == {
        "file_tool.path": str(target.resolve()),
        "file_tool.persistence": "history",
    }


@pytest.mark.asyncio
async def test_create_file_allows_empty_text(tmp_path: Path):
    target = tmp_path / "empty_created.txt"
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    result = await tool.call(_create_args(target, ""), "call_create_empty")

    assert target.read_text(encoding="utf-8") == ""
    assert result.output.startswith("CREATED ")
    assert "line_count: 0 → 0" in result.output
    assert "Added" not in result.output


@pytest.mark.asyncio
async def test_create_file_existing_path_returns_already_exists_live_window(tmp_path: Path):
    target = tmp_path / "exists.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    result = await tool.call(_create_args(target, "new\n"), "call_exists")

    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert result.output.startswith("ALREADY_EXISTS ")
    assert "1 | alpha" in result.output
    # ALREADY_EXISTS mints a diagnostic-source_kind window carrying the embedded current view.
    window = windows[-1]
    assert window.source_kind == "already_exists"
    assert window.status == "live"
    assert window.revision == _hash("alpha\nbeta\n")


@pytest.mark.asyncio
async def test_replace_lines_writes_disk_and_refreshes_prior_window(tmp_path: Path):
    target = tmp_path / "code.txt"
    initial = "one\ntwo\nthree\nfour\n"
    target.write_text(initial, encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    await tool.call(_read_args(target), "call_read")
    expected_revision = windows[-1].revision

    write_result = await tool.call(
        _replace_args(target, expected_revision, 2, 3, "TWO\nTHREE_X\n"),
        "call_write",
    )

    assert target.read_text(encoding="utf-8") == "one\nTWO\nTHREE_X\nfour\n"
    assert write_result.prokaryotes_annotations == {
        "file_tool.path": str(target.resolve()),
        "file_tool.persistence": "history",
    }
    assert write_result.output.startswith("EDITED ")
    assert "Removed (lines 2-3):" in write_result.output
    assert "Added (lines 2-3):" in write_result.output
    assert "line_count: 4 → 4" in write_result.output
    assert "Live windows refreshed for this path: 1." in write_result.output

    # The prior read window was refreshed in place at the new revision.
    read_window = next(w for w in windows if w.window_id == "call_read")
    assert read_window.status == "live"
    assert read_window.revision == _hash("one\nTWO\nTHREE_X\nfour\n")
    assert "1 | one" in read_window.rendered_output
    assert "2 | TWO" in read_window.rendered_output
    assert "3 | THREE_X" in read_window.rendered_output


@pytest.mark.asyncio
async def test_insert_lines_appends_at_eof(tmp_path: Path):
    target = tmp_path / "log.txt"
    target.write_text("a\nb\n", encoding="utf-8")
    rev = _hash("a\nb\n")
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    result = await tool.call(_insert_args(target, rev, 3, "c\nd\n"), "call_ins")

    assert target.read_text(encoding="utf-8") == "a\nb\nc\nd\n"
    assert "Added (lines 3-4):" in result.output
    assert "Removed" not in result.output
    assert "Live windows refreshed for this path: 0." in result.output


@pytest.mark.asyncio
async def test_delete_lines_only_emits_removed_block(tmp_path: Path):
    target = tmp_path / "data.txt"
    target.write_text("a\nb\nc\nd\n", encoding="utf-8")
    rev = _hash("a\nb\nc\nd\n")
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    result = await tool.call(_delete_args(target, rev, 2, 3), "call_del")

    assert target.read_text(encoding="utf-8") == "a\nd\n"
    assert "Removed (lines 2-3):" in result.output
    assert "Added" not in result.output


@pytest.mark.asyncio
async def test_replace_lines_emits_context_blocks_around_edit(tmp_path: Path):
    target = tmp_path / "code.txt"
    target.write_text("one\ntwo\nthree\nfour\nfive\nsix\n", encoding="utf-8")
    rev = _hash("one\ntwo\nthree\nfour\nfive\nsix\n")
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    result = await tool.call(
        _replace_args(target, rev, 3, 4, "THREE\nFOUR\n"),
        "call_replace_ctx",
    )

    assert "Removed (lines 3-4):" in result.output
    assert "Added (lines 3-4):" in result.output
    assert "Context before (lines 1-2):" in result.output
    assert "1 | one" in result.output
    assert "2 | two" in result.output
    assert "Context after (lines 5-6):" in result.output
    assert "5 | five" in result.output
    assert "6 | six" in result.output


@pytest.mark.asyncio
async def test_insert_lines_at_eof_emits_only_context_before(tmp_path: Path):
    target = tmp_path / "log.txt"
    target.write_text("a\nb\n", encoding="utf-8")
    rev = _hash("a\nb\n")
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    result = await tool.call(_insert_args(target, rev, 3, "c\nd\n"), "call_ins_eof")

    assert "Added (lines 3-4):" in result.output
    assert "Context before (lines 1-2):" in result.output
    assert "1 | a" in result.output
    assert "2 | b" in result.output
    assert "Context after" not in result.output


@pytest.mark.asyncio
async def test_delete_lines_emits_context_blocks_at_boundary(tmp_path: Path):
    target = tmp_path / "data.txt"
    target.write_text("a\nb\nc\nd\nE\nF\n", encoding="utf-8")
    rev = _hash("a\nb\nc\nd\nE\nF\n")
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    result = await tool.call(_delete_args(target, rev, 3, 4), "call_del_ctx")

    assert target.read_text(encoding="utf-8") == "a\nb\nE\nF\n"
    assert "Removed (lines 3-4):" in result.output
    assert "Context before (lines 1-2):" in result.output
    assert "1 | a" in result.output
    assert "2 | b" in result.output
    assert "Context after (lines 3-4):" in result.output
    assert "3 | E" in result.output
    assert "4 | F" in result.output


@pytest.mark.asyncio
async def test_replace_at_line_1_omits_context_before(tmp_path: Path):
    target = tmp_path / "code.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    rev = _hash("alpha\nbeta\ngamma\n")
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    result = await tool.call(
        _replace_args(target, rev, 1, 1, "ALPHA\n"),
        "call_replace_top",
    )

    assert "Added (lines 1-1):" in result.output
    assert "Context before" not in result.output
    assert "Context after (lines 2-3):" in result.output
    assert "2 | beta" in result.output
    assert "3 | gamma" in result.output


@pytest.mark.asyncio
async def test_replace_with_duplicate_trailing_line_shows_collision_in_context(tmp_path: Path):
    # Partition pathology: model rewrites a fenced block, closing its new_text with `` ``` `` while leaving the original
    # closing fence one line past end_line. The post-edit file then has two consecutive `` ``` `` lines, which Context
    # after surfaces inline.
    target = tmp_path / "doc.md"
    target.write_text("intro\n```\nbody\n```\noutro\n", encoding="utf-8")
    rev = _hash("intro\n```\nbody\n```\noutro\n")
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    # Replace lines 2-3 (opening fence + body) with a new fenced block that also closes the fence at its last line —
    # leaving the original closing fence on line 4.
    result = await tool.call(
        _replace_args(target, rev, 2, 3, "```\nNEW_BODY\n```\n"),
        "call_dup_fence",
    )

    assert "Added (lines 2-4):" in result.output
    assert "Context after (lines 5-6):" in result.output
    # The duplicate fence is now visible: Added ends at line 4 with ``` and the next line of the post-edit file (shown
    # by Context after) is also ```.
    assert "4 | ```" in result.output
    assert "5 | ```" in result.output


@pytest.mark.asyncio
async def test_write_with_stale_revision_returns_conflict_carrying_live_view(tmp_path: Path):
    target = tmp_path / "f.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    result = await tool.call(
        _replace_args(target, "wrong-revision", 1, 1, "ALPHA\n"),
        "call_conflict",
    )

    assert result.output.startswith("CONFLICT ")
    assert "Use the current view before retrying" in result.output
    window = windows[-1]
    assert window.source_kind == "conflict"
    assert window.status == "live"
    assert window.revision == _hash("alpha\nbeta\n")
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"


@pytest.mark.asyncio
async def test_write_with_out_of_range_returns_range_error_carrying_live_view(tmp_path: Path):
    target = tmp_path / "f.txt"
    target.write_text("a\nb\n", encoding="utf-8")
    rev = _hash("a\nb\n")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    result = await tool.call(
        _replace_args(target, rev, 5, 9, "X\n"),
        "call_range",
    )

    assert result.output.startswith("RANGE_ERROR ")
    window = windows[-1]
    assert window.source_kind == "range_error"
    assert window.status == "live"
    assert target.read_text(encoding="utf-8") == "a\nb\n"


@pytest.mark.asyncio
async def test_write_without_expected_revision_errors(tmp_path: Path):
    target = tmp_path / "f.txt"
    target.write_text("a\n", encoding="utf-8")
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    args = json.dumps(
        {
            "action": "delete_lines",
            "path": str(target),
            "expected_revision": None,
            "start_line": 1,
            "end_line": 1,
            "new_text": None,
        }
    )
    result = await tool.call(args, "call_no_rev")

    assert result.output.startswith("ERROR ")
    assert "expected_revision is required" in result.output
    assert target.read_text(encoding="utf-8") == "a\n"


@pytest.mark.asyncio
async def test_replace_and_insert_require_non_empty_new_text_string(tmp_path: Path):
    target = tmp_path / "f.txt"
    target.write_text("a\nb\n", encoding="utf-8")
    rev = _hash("a\nb\n")
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    replace_args = json.dumps(
        {
            "action": "replace_lines",
            "path": str(target),
            "expected_revision": rev,
            "start_line": 1,
            "end_line": 1,
            "new_text": None,
        }
    )
    insert_args = json.dumps(
        {
            "action": "insert_lines",
            "path": str(target),
            "expected_revision": rev,
            "start_line": 2,
            "end_line": None,
            "new_text": None,
        }
    )
    empty_replace_args = json.dumps(
        {
            "action": "replace_lines",
            "path": str(target),
            "expected_revision": rev,
            "start_line": 1,
            "end_line": 1,
            "new_text": "",
        }
    )

    replace_result = await tool.call(replace_args, "call_replace_null")
    insert_result = await tool.call(insert_args, "call_insert_null")
    empty_replace_result = await tool.call(empty_replace_args, "call_replace_empty")

    assert replace_result.output.startswith("ERROR ")
    assert "new_text is required for replace_lines" in replace_result.output
    assert insert_result.output.startswith("ERROR ")
    assert "new_text is required for insert_lines" in insert_result.output
    assert empty_replace_result.output.startswith("ERROR ")
    assert "new_text is required for replace_lines" in empty_replace_result.output
    assert target.read_text(encoding="utf-8") == "a\nb\n"


@pytest.mark.asyncio
async def test_create_file_requires_string_new_text_and_null_line_fields(tmp_path: Path):
    target = tmp_path / "new.txt"
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    bad_type_args = json.dumps(
        {
            "action": "create_file",
            "path": str(target),
            "expected_revision": None,
            "start_line": None,
            "end_line": None,
            "new_text": None,
        }
    )
    bad_start_args = json.dumps(
        {
            "action": "create_file",
            "path": str(target),
            "expected_revision": None,
            "start_line": 1,
            "end_line": None,
            "new_text": "alpha\n",
        }
    )

    bad_type_result = await tool.call(bad_type_args, "call_create_type")
    bad_start_result = await tool.call(bad_start_args, "call_create_start")

    assert bad_type_result.output.startswith("ERROR ")
    assert "new_text is required for create_file" in bad_type_result.output
    assert bad_start_result.output.startswith("ERROR ")
    assert "start_line must be null for create_file" in bad_start_result.output
    assert not target.exists()


@pytest.mark.asyncio
async def test_write_rejects_boolean_line_numbers(tmp_path: Path):
    target = tmp_path / "f.txt"
    target.write_text("a\nb\n", encoding="utf-8")
    rev = _hash("a\nb\n")
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    args = json.dumps(
        {
            "action": "delete_lines",
            "path": str(target),
            "expected_revision": rev,
            "start_line": True,
            "end_line": 1,
            "new_text": None,
        }
    )

    result = await tool.call(args, "call_bool")

    assert result.output.startswith("ERROR ")
    assert "start_line is required" in result.output
    assert result.prokaryotes_annotations is None
    assert target.read_text(encoding="utf-8") == "a\nb\n"


@pytest.mark.asyncio
async def test_read_rejects_files_over_max_file_bytes(tmp_path: Path, monkeypatch):
    target = tmp_path / "large.txt"
    target.write_text("01234567890", encoding="utf-8")
    monkeypatch.setattr(FileTool, "max_file_bytes", 10)
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    result = await tool.call(_read_args(target), "call_large")

    assert result.output.startswith("ERROR FileToolFileTooLargeError")
    assert result.prokaryotes_annotations is None


@pytest.mark.asyncio
async def test_reconcile_tombstones_live_window_when_file_grows_too_large(tmp_path: Path, monkeypatch):
    target = tmp_path / "grows.txt"
    target.write_text("small\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    await tool.call(_read_args(target), "call_read")

    target.write_text("01234567890", encoding="utf-8")
    monkeypatch.setattr(FileTool, "max_file_bytes", 10)

    await reconcile_working_files(
        windows,
        workspace_root=tmp_path,
        max_file_bytes=FileTool.max_file_bytes,
        max_lines=FileTool.max_lines,
    )

    tombstoned = windows[0]
    assert tombstoned.status == "stale"
    assert tombstoned.source_kind == "tombstone"
    assert "FileToolFileTooLargeError" in tombstoned.rendered_output


@pytest.mark.asyncio
async def test_write_rejects_edit_that_would_exceed_max_file_bytes(tmp_path: Path, monkeypatch):
    target = tmp_path / "grow_edit.txt"
    target.write_text("a\n", encoding="utf-8")
    rev = _hash("a\n")
    monkeypatch.setattr(FileTool, "max_file_bytes", 5)
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    result = await tool.call(_insert_args(target, rev, 2, "bbbb\n"), "call_grow")

    assert result.output.startswith("ERROR FileToolFileTooLargeError")
    assert result.prokaryotes_annotations is None
    assert target.read_text(encoding="utf-8") == "a\n"


@pytest.mark.parametrize(
    ("original", "action", "start", "end", "new_text", "expected"),
    [
        ("a\nb\n", "replace_lines", 2, 2, "B", "a\nB\n"),
        ("a\nb\n", "replace_lines", 2, 2, "B\n", "a\nB\n"),
        ("a\nb", "replace_lines", 2, 2, "B", "a\nB"),
        ("a\nb", "replace_lines", 2, 2, "B\n", "a\nB"),
        ("a\nb\n", "insert_lines", 2, None, "X", "a\nX\nb\n"),
        ("a\nb\n", "insert_lines", 2, None, "X\n", "a\nX\nb\n"),
        ("a\nb", "insert_lines", 2, None, "X", "a\nX\nb"),
        ("a\nb", "insert_lines", 2, None, "X\n", "a\nX\nb"),
        ("a\nb\n", "delete_lines", 2, 2, None, "a\n"),
        ("a\nb", "delete_lines", 2, 2, None, "a"),
        ("", "insert_lines", 1, None, "X", "X\n"),
        ("", "insert_lines", 1, None, "X\n", "X\n"),
    ],
)
@pytest.mark.asyncio
async def test_line_edits_preserve_existing_trailing_newline_policy(
    tmp_path: Path,
    original: str,
    action: str,
    start: int,
    end: int | None,
    new_text: str | None,
    expected: str,
):
    target = tmp_path / "newline.txt"
    target.write_text(original, encoding="utf-8")
    rev = _hash(original)
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    if action == "replace_lines":
        args = _replace_args(target, rev, start, end, new_text)
    elif action == "insert_lines":
        args = _insert_args(target, rev, start, new_text)
    else:
        args = _delete_args(target, rev, start, end)

    result = await tool.call(args, "call_newline")

    assert result.output.startswith("EDITED ")
    assert target.read_text(encoding="utf-8") == expected


@pytest.mark.asyncio
async def test_reconcile_working_files_refreshes_live_windows_after_external_edit(tmp_path: Path):
    target = tmp_path / "tracked.txt"
    target.write_text("v1\nv2\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    await tool.call(_read_args(target), "call_r")

    target.write_text("v1\nv2\nv3\n", encoding="utf-8")

    await reconcile_working_files(
        windows,
        workspace_root=tmp_path,
        max_file_bytes=FileTool.max_file_bytes,
        max_lines=FileTool.max_lines,
    )

    refreshed = windows[0]
    assert refreshed.revision == _hash("v1\nv2\nv3\n")
    assert "3 | v3" in refreshed.rendered_output
    assert refreshed.status == "live"


@pytest.mark.asyncio
async def test_reconcile_working_files_normalizes_conflict_window_without_revision_change(tmp_path: Path):
    target = tmp_path / "conflict.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    conflict_result = await tool.call(
        _replace_args(target, "stale-revision", 1, 1, "ALPHA\n"),
        "call_conflict",
    )
    assert conflict_result.output.startswith("CONFLICT ")
    assert windows[-1].source_kind == "conflict"

    await reconcile_working_files(
        windows,
        workspace_root=tmp_path,
        max_file_bytes=FileTool.max_file_bytes,
        max_lines=FileTool.max_lines,
    )

    normalized = windows[-1]
    assert normalized.source_kind == "read_lines"
    assert normalized.rendered_output.startswith("FILE ")
    assert "CONFLICT " not in normalized.rendered_output
    assert "1 | alpha" in normalized.rendered_output
    assert normalized.revision == _hash("alpha\nbeta\n")


@pytest.mark.asyncio
async def test_reconcile_working_files_normalizes_range_truncated_window_without_revision_change(tmp_path: Path):
    target = tmp_path / "truncated.txt"
    line_count = FileTool.max_lines + 10
    target.write_text("".join(f"line{i}\n" for i in range(1, line_count + 1)), encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    truncated_result = await tool.call(
        _read_args(target, start_line=1, end_line=line_count),
        "call_trunc",
    )
    assert truncated_result.output.startswith("RANGE_TRUNCATED ")
    assert windows[-1].source_kind == "range_truncated"

    await reconcile_working_files(
        windows,
        workspace_root=tmp_path,
        max_file_bytes=FileTool.max_file_bytes,
        max_lines=FileTool.max_lines,
    )

    normalized = windows[-1]
    assert normalized.source_kind == "read_lines"
    assert normalized.rendered_output.startswith("FILE ")
    assert "RANGE_TRUNCATED " not in normalized.rendered_output
    assert "1 | line1" in normalized.rendered_output
    assert f"{FileTool.max_lines} | line{FileTool.max_lines}" in normalized.rendered_output
    assert normalized.view_end_line == FileTool.max_lines
    assert normalized.requested_end_line == FileTool.max_lines


@pytest.mark.asyncio
async def test_reconcile_working_files_normalizes_already_exists_window_without_revision_change(tmp_path: Path):
    target = tmp_path / "exists_again.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    already_exists_result = await tool.call(_create_args(target, "ignored\n"), "call_exists")
    assert already_exists_result.output.startswith("ALREADY_EXISTS ")
    assert windows[-1].source_kind == "already_exists"

    await reconcile_working_files(
        windows,
        workspace_root=tmp_path,
        max_file_bytes=FileTool.max_file_bytes,
        max_lines=FileTool.max_lines,
    )

    normalized = windows[-1]
    assert normalized.source_kind == "read_lines"
    assert normalized.rendered_output.startswith("FILE ")
    assert "ALREADY_EXISTS " not in normalized.rendered_output
    assert "1 | alpha" in normalized.rendered_output
    assert normalized.revision == _hash("alpha\nbeta\n")


@pytest.mark.asyncio
async def test_read_refreshes_prior_live_windows_for_same_path(tmp_path: Path):
    # The second read must escape the prior window's intended coverage so it hits disk and refreshes — a covered re-read
    # short-circuits to REDUNDANT_READ instead.
    target = tmp_path / "read_refresh.txt"
    target.write_text("old1\nold2\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    first_args = _read_args(target, 1, 1)
    await tool.call(first_args, "call_first")

    target.write_text("new1\nnew2\n", encoding="utf-8")
    await tool.call(_read_args(target, 2, 2), "call_second")

    refreshed = next(w for w in windows if w.window_id == "call_first")
    new_rev = _hash("new1\nnew2\n")
    assert refreshed.revision == new_rev
    assert "1 | new1" in refreshed.rendered_output
    assert "1 | old1" not in refreshed.rendered_output
    second = next(w for w in windows if w.window_id == "call_second")
    assert second.revision == new_rev


@pytest.mark.asyncio
async def test_failed_read_tombstones_prior_live_windows_for_same_path(tmp_path: Path):
    # The follow-up read must escape the prior window's intended coverage so it reaches disk and discovers the missing
    # file — a covered re-read would short-circuit to REDUNDANT_READ without observing the deletion. Tombstone
    # discovery for the covered case shifts to the per-turn reconcile pass; that path is covered separately.
    target = tmp_path / "read_missing.txt"
    target.write_text("gone1\ngone2\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    await tool.call(_read_args(target, 1, 1), "call_read")

    target.unlink()
    missing_result = await tool.call(_read_args(target, 2, 2), "call_missing")

    tombstoned = next(w for w in windows if w.window_id == "call_read")
    assert missing_result.output.startswith("ERROR FileNotFoundError")
    assert tombstoned.status == "stale"
    assert tombstoned.source_kind == "tombstone"
    assert "no longer accessible" in tombstoned.rendered_output
    assert "FileNotFoundError" in tombstoned.rendered_output


@pytest.mark.asyncio
async def test_covered_exact_span_reread_returns_redundant_read(tmp_path: Path):
    target = tmp_path / "covered.txt"
    target.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    await tool.call(_read_args(target, 1, 3), "call_first")
    windows_before = len(windows)

    second_read = await tool.call(_read_args(target, 2, 3), "call_second")

    assert second_read.output.startswith("REDUNDANT_READ ")
    assert "rendered lines 1-3" in second_read.output
    assert "intended coverage 1-3" in second_read.output
    assert "page forward from start_line=4" in second_read.output
    assert second_read.prokaryotes_annotations == {
        "file_tool.path": str(target),
        "file_tool.persistence": "working_file",
    }
    # The short-circuit must not append a new working_file_window.
    assert len(windows) == windows_before


@pytest.mark.asyncio
async def test_covered_open_ended_reread_of_short_file_returns_redundant_read(tmp_path: Path):
    target = tmp_path / "short_open_ended.txt"
    target.write_text("a\nb\nc\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    await tool.call(_read_args(target), "call_first")
    first_window = windows[-1]
    assert first_window.view_end_line == 3
    assert first_window.requested_end_line is None

    second_read = await tool.call(_read_args(target), "call_second")

    assert second_read.output.startswith("REDUNDANT_READ ")
    assert "rendered lines 1-3" in second_read.output
    assert "intended coverage 1-200" in second_read.output
    assert "page forward from start_line=201" in second_read.output


@pytest.mark.asyncio
async def test_next_page_uses_intended_coverage_end_not_view_end(tmp_path: Path):
    target = tmp_path / "next_page.txt"
    target.write_text("a\nb\nc\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    await tool.call(_read_args(target), "call_first")

    # view_end_line + 1 = 4, small exact span inside intended coverage [1, 200].
    covered_exact = await tool.call(_read_args(target, 4, 4), "call_view_end_plus_one")
    assert covered_exact.output.startswith("REDUNDANT_READ ")
    assert "page forward from start_line=201" in covered_exact.output

    # intended_coverage_end + 1 = 201, escapes coverage.
    next_page = await tool.call(_read_args(target, 201), "call_intended_end_plus_one")
    assert not next_page.output.startswith("REDUNDANT_READ ")
    new_window = next(w for w in windows if w.window_id == "call_intended_end_plus_one")
    assert new_window.status == "live"


@pytest.mark.asyncio
async def test_range_truncated_window_counts_as_stable_coverage(tmp_path: Path):
    long_text = "".join(f"line{n}\n" for n in range(1, 301))
    target = tmp_path / "range_truncated.txt"
    target.write_text(long_text, encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    first_read = await tool.call(_read_args(target, 1, 250), "call_first")
    assert first_read.output.startswith("RANGE_TRUNCATED ")
    assert windows[-1].source_kind == "range_truncated"
    assert windows[-1].status == "live"

    second_read = await tool.call(_read_args(target, 1, 200), "call_second")
    assert second_read.output.startswith("REDUNDANT_READ ")


@pytest.mark.asyncio
async def test_already_exists_window_is_not_stable_coverage(tmp_path: Path):
    target = tmp_path / "already_exists.txt"
    target.write_text("a\nb\nc\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    create_result = await tool.call(_create_args(target, "ignored\n"), "call_create")
    assert create_result.output.startswith("ALREADY_EXISTS ")
    assert windows[-1].source_kind == "already_exists"

    follow_up = await tool.call(_read_args(target, 1, 3), "call_follow")
    assert not follow_up.output.startswith("REDUNDANT_READ ")


@pytest.mark.asyncio
async def test_conflict_window_is_not_stable_coverage(tmp_path: Path):
    target = tmp_path / "conflict.txt"
    target.write_text("a\nb\nc\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    stale_rev = _hash("not-the-real-content\n")
    conflict_result = await tool.call(_replace_args(target, stale_rev, 1, 1, "x\n"), "call_conflict")
    assert conflict_result.output.startswith("CONFLICT ")
    assert windows[-1].source_kind == "conflict"

    follow_up = await tool.call(_read_args(target, 1, 3), "call_follow")
    assert not follow_up.output.startswith("REDUNDANT_READ ")


@pytest.mark.asyncio
async def test_range_error_window_is_not_stable_coverage(tmp_path: Path):
    target = tmp_path / "range_error.txt"
    target.write_text("a\nb\nc\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    rev = _hash("a\nb\nc\n")
    range_error_result = await tool.call(_replace_args(target, rev, 50, 60, "x\n"), "call_range_error")
    assert range_error_result.output.startswith("RANGE_ERROR ")
    assert windows[-1].source_kind == "range_error"
    assert windows[-1].view_start_line == 50

    # If RANGE_ERROR were stable coverage, this read of [50, 60] would short-circuit. It must not.
    follow_up = await tool.call(_read_args(target, 50, 60), "call_follow")
    assert not follow_up.output.startswith("REDUNDANT_READ ")


@pytest.mark.asyncio
async def test_redundant_read_does_not_mint_window_and_does_not_block_refresh_or_tombstone(tmp_path: Path):
    target = tmp_path / "redundant_annotations.txt"
    target.write_text("a\nb\nc\nd\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    await tool.call(_read_args(target, 1, 4), "call_first")
    initial_window_count = len(windows)

    redundant = await tool.call(_read_args(target, 2, 3), "call_redundant")
    assert redundant.output.startswith("REDUNDANT_READ ")
    assert redundant.prokaryotes_annotations == {
        "file_tool.path": str(target),
        "file_tool.persistence": "working_file",
    }
    # REDUNDANT_READ does not mint a new window.
    assert len(windows) == initial_window_count

    # A follow-up edit refreshes exactly one live window — the original read.
    first_window = windows[0]
    edit_result = await tool.call(_replace_args(target, first_window.revision, 1, 1, "A\n"), "call_edit")
    assert edit_result.output.startswith("EDITED ")
    assert "Live windows refreshed for this path: 1" in edit_result.output

    # Tombstone on missing file affects only the live windows for the path.
    target.unlink()
    await tool.call(_read_args(target, 50, 60), "call_missing")
    assert windows[0].status == "stale"
    assert windows[0].source_kind == "tombstone"


@pytest.mark.asyncio
async def test_failed_write_tombstones_prior_live_windows_for_same_path(tmp_path: Path):
    target = tmp_path / "write_missing.txt"
    target.write_text("gone soon\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    await tool.call(_read_args(target), "call_read")
    rev = windows[-1].revision

    target.unlink()
    write_result = await tool.call(_replace_args(target, rev, 1, 1, "replacement\n"), "call_write")

    tombstoned = next(w for w in windows if w.window_id == "call_read")
    assert write_result.output.startswith("ERROR FileNotFoundError")
    assert tombstoned.status == "stale"
    assert tombstoned.source_kind == "tombstone"
    assert "no longer accessible" in tombstoned.rendered_output
    assert "FileNotFoundError" in tombstoned.rendered_output


@pytest.mark.asyncio
async def test_write_refreshes_same_round_read_window(tmp_path: Path):
    """A read in the same turn mints a window; a subsequent write at the read's revision refreshes that window
    in place. Replaces the previous test that asserted on the TurnItem's annotation — same behavior, working
    against `WorkingFileWindow` state."""
    target = tmp_path / "same_round.txt"
    target.write_text("before\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    await tool.call(_read_args(target), "call_read")
    write_result = await tool.call(
        _replace_args(target, _hash("before\n"), 1, 1, "after\n"),
        "call_write",
    )

    assert write_result.output.startswith("EDITED ")
    read_window = next(w for w in windows if w.window_id == "call_read")
    assert read_window.revision == _hash("after\n")
    assert "1 | after" in read_window.rendered_output
    assert "1 | before" not in read_window.rendered_output


@pytest.mark.asyncio
async def test_reconcile_working_files_tombstones_when_path_disappears(tmp_path: Path):
    target = tmp_path / "vanishing.txt"
    target.write_text("here today\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    await tool.call(_read_args(target), "call_t")

    target.unlink()
    await reconcile_working_files(
        windows,
        workspace_root=tmp_path,
        max_file_bytes=FileTool.max_file_bytes,
        max_lines=FileTool.max_lines,
    )

    tombstoned = windows[0]
    assert tombstoned.status == "stale"
    assert tombstoned.source_kind == "tombstone"
    assert "no longer accessible" in tombstoned.rendered_output
    assert "FileNotFoundError" in tombstoned.rendered_output


@pytest.mark.asyncio
async def test_reconcile_working_files_tombstones_when_path_now_escapes_workspace(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    target = workspace / "tracked.txt"
    outside = outside_dir / "secret.txt"
    target.write_text("inside\n", encoding="utf-8")
    outside.write_text("outside\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=workspace)

    await tool.call(_read_args(target), "call_escape")

    target.unlink()
    target.symlink_to(outside)
    await reconcile_working_files(
        windows,
        workspace_root=workspace,
        max_file_bytes=FileTool.max_file_bytes,
        max_lines=FileTool.max_lines,
    )

    tombstoned = windows[0]
    assert tombstoned.status == "stale"
    assert tombstoned.source_kind == "tombstone"
    assert "no longer accessible" in tombstoned.rendered_output
    assert "ValueError" in tombstoned.rendered_output
    assert "outside" not in tombstoned.rendered_output


@pytest.mark.asyncio
async def test_reconcile_working_files_skips_items_already_at_current_revision(tmp_path: Path):
    target = tmp_path / "stable.txt"
    target.write_text("unchanged\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    await tool.call(_read_args(target), "call_s")
    output_before = windows[0].rendered_output

    await reconcile_working_files(
        windows,
        workspace_root=tmp_path,
        max_file_bytes=FileTool.max_file_bytes,
        max_lines=FileTool.max_lines,
    )

    assert windows[0].rendered_output == output_before


def test_refresh_windows_for_path_handles_multiple_views_into_same_path():
    text_v1 = "a\nb\nc\nd\ne\n"
    text_v2 = "A\nB\nC\nD\nE\n"
    rev_v1 = _hash(text_v1)
    rev_v2 = _hash(text_v2)
    win_first = WorkingFileWindow(
        window_id="c1",
        path="/tmp/x",
        status="live",
        revision=rev_v1,
        rendered_output="placeholder",
        view_start_line=1,
        view_end_line=3,
        requested_end_line=None,
        source_kind="read_lines",
    )
    win_second = WorkingFileWindow(
        window_id="c2",
        path="/tmp/x",
        status="live",
        revision=rev_v1,
        rendered_output="placeholder",
        view_start_line=3,
        view_end_line=5,
        requested_end_line=None,
        source_kind="read_lines",
    )

    refreshed_count = refresh_windows_for_path([win_first, win_second], "/tmp/x", text_v2, rev_v2, FileTool.max_lines)

    assert refreshed_count == 2
    assert win_first.revision == rev_v2
    assert "1 | A" in win_first.rendered_output
    assert win_second.revision == rev_v2
    assert "3 | C" in win_second.rendered_output


def test_render_view_returns_empty_view_past_eof():
    end_line, line_count, view_lines = render_view("a\nb\n", start_line=10, max_lines=5)

    assert line_count == 2
    assert view_lines == []
    assert end_line == 9


def test_render_view_honors_requested_end_line():
    end_line, line_count, view_lines = render_view(
        "a\nb\nc\nd\n",
        start_line=2,
        max_lines=5,
        requested_end_line=3,
    )

    assert line_count == 4
    assert view_lines == ["b", "c"]
    assert end_line == 3


def test_refresh_windows_for_path_preserves_exact_requested_range():
    text_v1 = "a\nb\nc\nd\ne\n"
    text_v2 = "A\nB\nC\nD\nE\nF\n"
    rev_v1 = _hash(text_v1)
    rev_v2 = _hash(text_v2)
    window = WorkingFileWindow(
        window_id="c1",
        path="/tmp/x",
        status="live",
        revision=rev_v1,
        rendered_output="placeholder",
        view_start_line=2,
        view_end_line=3,
        requested_end_line=3,
        source_kind="read_lines",
    )

    refreshed_count = refresh_windows_for_path([window], "/tmp/x", text_v2, rev_v2, FileTool.max_lines)

    assert refreshed_count == 1
    assert window.revision == rev_v2
    assert window.view_end_line == 3
    assert "2 | B" in window.rendered_output
    assert "3 | C" in window.rendered_output
    assert "4 | D" not in window.rendered_output


def test_refresh_windows_for_path_returns_zero_for_already_current_items():
    text = "a\nb\n"
    rev = _hash(text)
    window = WorkingFileWindow(
        window_id="c1",
        path="/tmp/x",
        status="live",
        revision=rev,
        rendered_output="original",
        view_start_line=1,
        view_end_line=2,
        requested_end_line=None,
        source_kind="read_lines",
    )

    refreshed_count = refresh_windows_for_path([window], "/tmp/x", text, rev, FileTool.max_lines)

    assert refreshed_count == 0
    assert window.rendered_output == "original"


@pytest.mark.asyncio
async def test_concurrent_writes_same_path_yield_one_edit_one_conflict(tmp_path: Path):
    target = tmp_path / "shared.txt"
    target.write_text("x\n", encoding="utf-8")
    rev_a = _hash("x\n")
    windows_a: list[WorkingFileWindow] = []
    windows_b: list[WorkingFileWindow] = []
    tool_a = FileTool(working_file_provider=lambda: windows_a, workspace_root=tmp_path)
    tool_b = FileTool(working_file_provider=lambda: windows_b, workspace_root=tmp_path)

    result_a, result_b = await asyncio.gather(
        tool_a.call(_replace_args(target, rev_a, 1, 1, "A\n"), "call_a"),
        tool_b.call(_replace_args(target, rev_a, 1, 1, "B\n"), "call_b"),
    )

    outputs = (result_a.output, result_b.output)
    assert sum(o.startswith("EDITED ") for o in outputs) == 1
    assert sum(o.startswith("CONFLICT ") for o in outputs) == 1
    final = target.read_text(encoding="utf-8")
    assert final in {"A\n", "B\n"}
    # Whichever tool ran the CONFLICT minted a diagnostic window at the current revision.
    conflict_windows = windows_a if result_a.output.startswith("CONFLICT ") else windows_b
    conflict_window = conflict_windows[-1]
    assert conflict_window.source_kind == "conflict"
    assert conflict_window.revision == _hash(final)


@pytest.mark.asyncio
async def test_concurrent_writes_different_paths_do_not_block_each_other(tmp_path: Path):
    target_a = tmp_path / "a.txt"
    target_b = tmp_path / "b.txt"
    target_a.write_text("a\n", encoding="utf-8")
    target_b.write_text("b\n", encoding="utf-8")
    tool_a = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)
    tool_b = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    result_a, result_b = await asyncio.gather(
        tool_a.call(_replace_args(target_a, _hash("a\n"), 1, 1, "A\n"), "call_a"),
        tool_b.call(_replace_args(target_b, _hash("b\n"), 1, 1, "B\n"), "call_b"),
    )

    assert result_a.output.startswith("EDITED ")
    assert result_b.output.startswith("EDITED ")
    assert target_a.read_text(encoding="utf-8") == "A\n"
    assert target_b.read_text(encoding="utf-8") == "B\n"


@pytest.mark.asyncio
async def test_get_path_lock_returns_same_instance_for_same_path(tmp_path: Path):
    path_one = str(tmp_path / "one.txt")
    path_two = str(tmp_path / "two.txt")
    lock_one_a = reads._get_path_lock(path_one)
    lock_one_b = reads._get_path_lock(path_one)
    lock_two = reads._get_path_lock(path_two)

    assert lock_one_a is lock_one_b
    assert lock_two is not lock_one_a


@pytest.mark.asyncio
async def test_conflict_refreshes_prior_live_windows_for_same_path(tmp_path: Path):
    target = tmp_path / "drift.txt"
    target.write_text("old1\nold2\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)

    await tool.call(_read_args(target), "call_read")
    rev_a = windows[-1].revision

    target.write_text("new1\nnew2\nnew3\n", encoding="utf-8")
    rev_b = _hash("new1\nnew2\nnew3\n")

    write_result = await tool.call(
        _replace_args(target, rev_a, 1, 1, "X\n"),
        "call_write",
    )

    assert write_result.output.startswith("CONFLICT ")
    conflict_window = next(w for w in windows if w.window_id == "call_write")
    assert conflict_window.revision == rev_b
    assert conflict_window.source_kind == "conflict"
    refreshed = next(w for w in windows if w.window_id == "call_read")
    assert refreshed.revision == rev_b
    assert refreshed.status == "live"
    assert "1 | new1" in refreshed.rendered_output
    assert "3 | new3" in refreshed.rendered_output


@pytest.mark.asyncio
async def test_flock_alone_serializes_concurrent_locked_write_transactions(tmp_path: Path):
    target = tmp_path / "flock_target.txt"
    target.write_text("v0\n", encoding="utf-8")
    rev = _hash("v0\n")
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    args_a = {
        "action": "replace_lines",
        "path": str(target),
        "expected_revision": rev,
        "start_line": 1,
        "end_line": 1,
        "new_text": "vA\n",
    }
    args_b = {
        "action": "replace_lines",
        "path": str(target),
        "expected_revision": rev,
        "start_line": 1,
        "end_line": 1,
        "new_text": "vB\n",
    }

    result_a, result_b = await asyncio.gather(
        asyncio.to_thread(
            tool._locked_write_transaction,
            "call_a",
            target.resolve(),
            "replace_lines",
            args_a,
            rev,
        ),
        asyncio.to_thread(
            tool._locked_write_transaction,
            "call_b",
            target.resolve(),
            "replace_lines",
            args_b,
            rev,
        ),
    )

    items = (result_a[0], result_b[0])
    assert sum(item.output.startswith("EDITED ") for item in items) == 1
    assert sum(item.output.startswith("CONFLICT ") for item in items) == 1
    final = target.read_text(encoding="utf-8")
    assert final in {"vA\n", "vB\n"}


@pytest.mark.asyncio
async def test_read_waits_for_same_path_lock_before_snapshotting(tmp_path: Path):
    target = tmp_path / "locked_read.txt"
    target.write_text("initial\n", encoding="utf-8")
    windows: list[WorkingFileWindow] = []
    tool = FileTool(working_file_provider=lambda: windows, workspace_root=tmp_path)
    path_lock = reads._get_path_lock(str(target.resolve()))

    async with path_lock:
        target.write_text("partial\n", encoding="utf-8")
        read_task = asyncio.create_task(tool.call(_read_args(target), "call_read"))
        await asyncio.sleep(0)
        assert not read_task.done()
        target.write_text("final\n", encoding="utf-8")

    result = await asyncio.wait_for(read_task, timeout=5)
    assert "1 | final" in result.output
    assert "partial" not in result.output
    assert windows[-1].revision == _hash("final\n")


@pytest.mark.asyncio
async def test_locked_read_text_waits_for_exclusive_flock(tmp_path: Path):
    target = tmp_path / "flocked_read.txt"
    target.write_text("initial\n", encoding="utf-8")
    writer_locked = threading.Event()
    release_writer = threading.Event()

    def writer_holding_flock():
        with open(target, "r+", encoding="utf-8") as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
            fp.seek(0)
            fp.truncate()
            fp.write("partial\n")
            fp.flush()
            writer_locked.set()
            assert release_writer.wait(timeout=5)
            fp.seek(0)
            fp.truncate()
            fp.write("final\n")
            fp.flush()

    thread = threading.Thread(target=writer_holding_flock)
    thread.start()
    assert writer_locked.wait(timeout=5)

    read_task = asyncio.create_task(asyncio.to_thread(_locked_read_text, target, 10_000_000))
    await asyncio.sleep(0.05)
    assert not read_task.done()
    release_writer.set()

    text = await asyncio.wait_for(read_task, timeout=5)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert text == "final\n"
