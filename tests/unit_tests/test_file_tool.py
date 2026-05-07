import asyncio
import fcntl
import json
import threading
from hashlib import sha256
from pathlib import Path

import pytest

from prokaryotes.api_v1.models import ContextPartition, ContextPartitionItem
from prokaryotes.tools_v1.file_tool import (
    FileTool,
    _locked_read_text,
    _refresh_live_windows,
    reconcile_tracked_files,
    render_view,
)


def _empty_partition() -> ContextPartition:
    return ContextPartition(conversation_uuid="conv", items=[])


def _hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _read_args(path: Path, start_line: int | None = None) -> str:
    return json.dumps({
        "action": "read",
        "path": str(path),
        "expected_revision": None,
        "start_line": start_line,
        "end_line": None,
        "new_text": None,
    })


def _replace_args(path: Path, expected_revision: str, start: int, end: int, new_text: str) -> str:
    return json.dumps({
        "action": "replace_lines",
        "path": str(path),
        "expected_revision": expected_revision,
        "start_line": start,
        "end_line": end,
        "new_text": new_text,
    })


def _create_args(path: Path, new_text: str) -> str:
    return json.dumps({
        "action": "create_file",
        "path": str(path),
        "expected_revision": None,
        "start_line": None,
        "end_line": None,
        "new_text": new_text,
    })


def _insert_args(path: Path, expected_revision: str, start: int, new_text: str) -> str:
    return json.dumps({
        "action": "insert_lines",
        "path": str(path),
        "expected_revision": expected_revision,
        "start_line": start,
        "end_line": None,
        "new_text": new_text,
    })


def _delete_args(path: Path, expected_revision: str, start: int, end: int) -> str:
    return json.dumps({
        "action": "delete_lines",
        "path": str(path),
        "expected_revision": expected_revision,
        "start_line": start,
        "end_line": end,
        "new_text": None,
    })


@pytest.mark.asyncio
async def test_read_returns_live_window_with_annotations(tmp_path: Path):
    target = tmp_path / "hello.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    result = await tool.call(_read_args(target), "call_1")

    assert result.type == "function_call_output"
    assert result.call_id == "call_1"
    assert result.prokaryotes_annotations["file_tool.path"] == str(target.resolve())
    assert result.prokaryotes_annotations["file_tool.status"] == "live"
    assert result.prokaryotes_annotations["file_tool.view_start_line"] == "1"
    assert result.prokaryotes_annotations["file_tool.view_end_line"] == "3"
    assert result.prokaryotes_annotations["file_tool.revision"] == _hash("alpha\nbeta\ngamma\n")
    assert "1 | alpha" in result.output
    assert "line_count=3" in result.output


@pytest.mark.asyncio
async def test_read_empty_file_yields_zero_line_count(tmp_path: Path):
    target = tmp_path / "empty.txt"
    target.write_text("", encoding="utf-8")
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    result = await tool.call(_read_args(target), "call_e")

    assert result.prokaryotes_annotations["file_tool.status"] == "live"
    assert "line_count=0" in result.output
    assert "1 | " not in result.output


@pytest.mark.asyncio
async def test_read_missing_file_returns_error(tmp_path: Path):
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    result = await tool.call(_read_args(tmp_path / "missing.txt"), "call_x")

    assert result.output.startswith("ERROR FileNotFoundError")
    assert result.prokaryotes_annotations is None


@pytest.mark.asyncio
async def test_read_path_escape_returns_error(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope\n", encoding="utf-8")
    tool = FileTool(_empty_partition(), workspace_root=workspace)

    result = await tool.call(_read_args(outside), "call_esc")

    assert result.output.startswith("ERROR ValueError")
    assert "escapes workspace root" in result.output


@pytest.mark.asyncio
async def test_read_rejects_non_positive_start_line(tmp_path: Path):
    target = tmp_path / "hello.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    zero_result = await tool.call(_read_args(target, start_line=0), "call_zero")
    negative_result = await tool.call(_read_args(target, start_line=-3), "call_neg")

    assert zero_result.output.startswith("ERROR ValueError")
    assert "start_line for read" in zero_result.output
    assert zero_result.prokaryotes_annotations is None
    assert negative_result.output.startswith("ERROR ValueError")
    assert "start_line for read" in negative_result.output
    assert negative_result.prokaryotes_annotations is None


@pytest.mark.asyncio
async def test_default_workspace_root_is_current_working_directory(tmp_path: Path, monkeypatch):
    target = tmp_path / "relative.txt"
    target.write_text("from cwd\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    tool = FileTool(_empty_partition())

    result = await tool.call(_read_args(Path("relative.txt")), "call_rel")

    assert result.prokaryotes_annotations["file_tool.path"] == str(target.resolve())
    assert "1 | from cwd" in result.output


@pytest.mark.asyncio
async def test_create_file_writes_new_file_and_returns_created_record(tmp_path: Path):
    target = tmp_path / "created.txt"
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    result = await tool.call(_create_args(target, "alpha\nbeta\n"), "call_create")

    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert result.prokaryotes_annotations == {"file_tool.path": str(target.resolve())}
    assert result.output.startswith("CREATED ")
    assert "revision: " in result.output
    assert "line_count: 0 → 2" in result.output
    assert "Added (lines 1-2):" in result.output


@pytest.mark.asyncio
async def test_create_file_creates_missing_parent_directories(tmp_path: Path):
    target = tmp_path / "nested" / "deeper" / "created.txt"
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    result = await tool.call(_create_args(target, "alpha\n"), "call_create_nested")

    assert target.read_text(encoding="utf-8") == "alpha\n"
    assert target.parent.is_dir()
    assert result.output.startswith("CREATED ")
    assert result.prokaryotes_annotations == {"file_tool.path": str(target.resolve())}


@pytest.mark.asyncio
async def test_create_file_allows_empty_text(tmp_path: Path):
    target = tmp_path / "empty_created.txt"
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    result = await tool.call(_create_args(target, ""), "call_create_empty")

    assert target.read_text(encoding="utf-8") == ""
    assert result.output.startswith("CREATED ")
    assert "line_count: 0 → 0" in result.output
    assert "Added" not in result.output


@pytest.mark.asyncio
async def test_create_file_existing_path_returns_already_exists_live_window(tmp_path: Path):
    target = tmp_path / "exists.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    result = await tool.call(_create_args(target, "new\n"), "call_exists")

    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert result.output.startswith("ALREADY_EXISTS ")
    assert result.prokaryotes_annotations["file_tool.status"] == "live"
    assert result.prokaryotes_annotations["file_tool.revision"] == _hash("alpha\nbeta\n")
    assert "1 | alpha" in result.output


@pytest.mark.asyncio
async def test_replace_lines_writes_disk_and_refreshes_prior_window(tmp_path: Path):
    target = tmp_path / "code.txt"
    initial = "one\ntwo\nthree\nfour\n"
    target.write_text(initial, encoding="utf-8")
    partition = _empty_partition()
    tool = FileTool(partition, workspace_root=tmp_path)

    read_result = await tool.call(_read_args(target), "call_read")
    partition.append(ContextPartitionItem(
        call_id="call_read",
        name="file_tool",
        arguments=_read_args(target),
        type="function_call",
    ))
    partition.append(read_result)

    expected_revision = read_result.prokaryotes_annotations["file_tool.revision"]
    write_result = await tool.call(
        _replace_args(target, expected_revision, 2, 3, "TWO\nTHREE_X\n"),
        "call_write",
    )

    assert target.read_text(encoding="utf-8") == "one\nTWO\nTHREE_X\nfour\n"
    assert write_result.prokaryotes_annotations == {"file_tool.path": str(target.resolve())}
    assert write_result.output.startswith("EDITED ")
    assert "Removed (lines 2-3):" in write_result.output
    assert "Added (lines 2-3):" in write_result.output
    assert "line_count: 4 → 4" in write_result.output

    refreshed = partition.items[1]
    assert refreshed.prokaryotes_annotations["file_tool.status"] == "live"
    assert refreshed.prokaryotes_annotations["file_tool.revision"] == _hash("one\nTWO\nTHREE_X\nfour\n")
    assert "1 | one" in refreshed.output
    assert "2 | TWO" in refreshed.output
    assert "3 | THREE_X" in refreshed.output


@pytest.mark.asyncio
async def test_insert_lines_appends_at_eof(tmp_path: Path):
    target = tmp_path / "log.txt"
    target.write_text("a\nb\n", encoding="utf-8")
    rev = _hash("a\nb\n")
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    result = await tool.call(_insert_args(target, rev, 3, "c\nd\n"), "call_ins")

    assert target.read_text(encoding="utf-8") == "a\nb\nc\nd\n"
    assert "Added (lines 3-4):" in result.output
    assert "Removed" not in result.output


@pytest.mark.asyncio
async def test_delete_lines_only_emits_removed_block(tmp_path: Path):
    target = tmp_path / "data.txt"
    target.write_text("a\nb\nc\nd\n", encoding="utf-8")
    rev = _hash("a\nb\nc\nd\n")
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    result = await tool.call(_delete_args(target, rev, 2, 3), "call_del")

    assert target.read_text(encoding="utf-8") == "a\nd\n"
    assert "Removed (lines 2-3):" in result.output
    assert "Added" not in result.output


@pytest.mark.asyncio
async def test_write_with_stale_revision_returns_conflict_carrying_live_view(tmp_path: Path):
    target = tmp_path / "f.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    result = await tool.call(
        _replace_args(target, "wrong-revision", 1, 1, "ALPHA\n"),
        "call_conflict",
    )

    assert result.output.startswith("CONFLICT ")
    assert "Re-read before retrying" in result.output
    assert result.prokaryotes_annotations["file_tool.status"] == "live"
    assert result.prokaryotes_annotations["file_tool.revision"] == _hash("alpha\nbeta\n")
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"


@pytest.mark.asyncio
async def test_write_with_out_of_range_returns_range_error_carrying_live_view(tmp_path: Path):
    target = tmp_path / "f.txt"
    target.write_text("a\nb\n", encoding="utf-8")
    rev = _hash("a\nb\n")
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    result = await tool.call(
        _replace_args(target, rev, 5, 9, "X\n"),
        "call_range",
    )

    assert result.output.startswith("RANGE_ERROR ")
    assert result.prokaryotes_annotations["file_tool.status"] == "live"
    assert target.read_text(encoding="utf-8") == "a\nb\n"


@pytest.mark.asyncio
async def test_write_without_expected_revision_errors(tmp_path: Path):
    target = tmp_path / "f.txt"
    target.write_text("a\n", encoding="utf-8")
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    args = json.dumps({
        "action": "delete_lines",
        "path": str(target),
        "expected_revision": None,
        "start_line": 1,
        "end_line": 1,
        "new_text": None,
    })
    result = await tool.call(args, "call_no_rev")

    assert result.output.startswith("ERROR ")
    assert "expected_revision is required" in result.output
    assert target.read_text(encoding="utf-8") == "a\n"


@pytest.mark.asyncio
async def test_replace_and_insert_require_non_empty_new_text_string(tmp_path: Path):
    target = tmp_path / "f.txt"
    target.write_text("a\nb\n", encoding="utf-8")
    rev = _hash("a\nb\n")
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    replace_args = json.dumps({
        "action": "replace_lines",
        "path": str(target),
        "expected_revision": rev,
        "start_line": 1,
        "end_line": 1,
        "new_text": None,
    })
    insert_args = json.dumps({
        "action": "insert_lines",
        "path": str(target),
        "expected_revision": rev,
        "start_line": 2,
        "end_line": None,
        "new_text": None,
    })
    empty_replace_args = json.dumps({
        "action": "replace_lines",
        "path": str(target),
        "expected_revision": rev,
        "start_line": 1,
        "end_line": 1,
        "new_text": "",
    })

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
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    bad_type_args = json.dumps({
        "action": "create_file",
        "path": str(target),
        "expected_revision": None,
        "start_line": None,
        "end_line": None,
        "new_text": None,
    })
    bad_start_args = json.dumps({
        "action": "create_file",
        "path": str(target),
        "expected_revision": None,
        "start_line": 1,
        "end_line": None,
        "new_text": "alpha\n",
    })

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
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    args = json.dumps({
        "action": "delete_lines",
        "path": str(target),
        "expected_revision": rev,
        "start_line": True,
        "end_line": 1,
        "new_text": None,
    })

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
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    result = await tool.call(_read_args(target), "call_large")

    assert result.output.startswith("ERROR FileToolFileTooLargeError")
    assert result.prokaryotes_annotations is None


@pytest.mark.asyncio
async def test_reconcile_tombstones_live_window_when_file_grows_too_large(tmp_path: Path, monkeypatch):
    target = tmp_path / "grows.txt"
    target.write_text("small\n", encoding="utf-8")
    partition = _empty_partition()
    tool = FileTool(partition, workspace_root=tmp_path)

    read_result = await tool.call(_read_args(target), "call_read")
    partition.append(ContextPartitionItem(
        call_id="call_read",
        name="file_tool",
        arguments=_read_args(target),
        type="function_call",
    ))
    partition.append(read_result)

    target.write_text("01234567890", encoding="utf-8")
    monkeypatch.setattr(FileTool, "max_file_bytes", 10)

    await reconcile_tracked_files(partition, workspace_root=tmp_path)

    tombstoned = partition.items[1]
    assert tombstoned.prokaryotes_annotations["file_tool.status"] == "stale"
    assert "FileToolFileTooLargeError" in tombstoned.output


@pytest.mark.asyncio
async def test_write_rejects_edit_that_would_exceed_max_file_bytes(tmp_path: Path, monkeypatch):
    target = tmp_path / "grow_edit.txt"
    target.write_text("a\n", encoding="utf-8")
    rev = _hash("a\n")
    monkeypatch.setattr(FileTool, "max_file_bytes", 5)
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

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
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

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
async def test_reconcile_tracked_files_refreshes_live_windows_after_external_edit(tmp_path: Path):
    target = tmp_path / "tracked.txt"
    target.write_text("v1\nv2\n", encoding="utf-8")
    partition = _empty_partition()
    tool = FileTool(partition, workspace_root=tmp_path)

    read_result = await tool.call(_read_args(target), "call_r")
    partition.append(ContextPartitionItem(
        call_id="call_r",
        name="file_tool",
        arguments=_read_args(target),
        type="function_call",
    ))
    partition.append(read_result)

    target.write_text("v1\nv2\nv3\n", encoding="utf-8")

    await reconcile_tracked_files(partition, workspace_root=tmp_path)

    refreshed = partition.items[1]
    assert refreshed.prokaryotes_annotations["file_tool.revision"] == _hash("v1\nv2\nv3\n")
    assert "3 | v3" in refreshed.output
    assert refreshed.prokaryotes_annotations["file_tool.status"] == "live"


@pytest.mark.asyncio
async def test_reconcile_tracked_files_normalizes_conflict_window_without_revision_change(tmp_path: Path):
    target = tmp_path / "conflict.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    partition = _empty_partition()
    tool = FileTool(partition, workspace_root=tmp_path)

    conflict_result = await tool.call(
        _replace_args(target, "stale-revision", 1, 1, "ALPHA\n"),
        "call_conflict",
    )
    partition.append(ContextPartitionItem(
        call_id="call_conflict",
        name="file_tool",
        arguments=_replace_args(target, "stale-revision", 1, 1, "ALPHA\n"),
        type="function_call",
    ))
    partition.append(conflict_result)
    assert conflict_result.output.startswith("CONFLICT ")

    await reconcile_tracked_files(partition, workspace_root=tmp_path)

    normalized = partition.items[1]
    assert normalized.output.startswith("FILE ")
    assert "CONFLICT " not in normalized.output
    assert "1 | alpha" in normalized.output
    assert normalized.prokaryotes_annotations["file_tool.revision"] == _hash("alpha\nbeta\n")


@pytest.mark.asyncio
async def test_reconcile_tracked_files_normalizes_already_exists_window_without_revision_change(tmp_path: Path):
    target = tmp_path / "exists_again.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    partition = _empty_partition()
    tool = FileTool(partition, workspace_root=tmp_path)

    already_exists_result = await tool.call(_create_args(target, "ignored\n"), "call_exists")
    partition.append(ContextPartitionItem(
        call_id="call_exists",
        name="file_tool",
        arguments=_create_args(target, "ignored\n"),
        type="function_call",
    ))
    partition.append(already_exists_result)
    assert already_exists_result.output.startswith("ALREADY_EXISTS ")

    await reconcile_tracked_files(partition, workspace_root=tmp_path)

    normalized = partition.items[1]
    assert normalized.output.startswith("FILE ")
    assert "ALREADY_EXISTS " not in normalized.output
    assert "1 | alpha" in normalized.output
    assert normalized.prokaryotes_annotations["file_tool.revision"] == _hash("alpha\nbeta\n")


@pytest.mark.asyncio
async def test_read_refreshes_prior_live_windows_for_same_path(tmp_path: Path):
    target = tmp_path / "read_refresh.txt"
    target.write_text("old\n", encoding="utf-8")
    partition = _empty_partition()
    tool = FileTool(partition, workspace_root=tmp_path)

    first_read = await tool.call(_read_args(target), "call_first")
    partition.append(ContextPartitionItem(
        call_id="call_first",
        name="file_tool",
        arguments=_read_args(target),
        type="function_call",
    ))
    partition.append(first_read)

    target.write_text("new\n", encoding="utf-8")
    second_read = await tool.call(_read_args(target), "call_second")

    refreshed = partition.items[1]
    assert refreshed.prokaryotes_annotations["file_tool.revision"] == _hash("new\n")
    assert "1 | new" in refreshed.output
    assert "1 | old" not in refreshed.output
    assert second_read.prokaryotes_annotations["file_tool.revision"] == _hash("new\n")


@pytest.mark.asyncio
async def test_failed_read_tombstones_prior_live_windows_for_same_path(tmp_path: Path):
    target = tmp_path / "read_missing.txt"
    target.write_text("gone soon\n", encoding="utf-8")
    partition = _empty_partition()
    tool = FileTool(partition, workspace_root=tmp_path)

    read_result = await tool.call(_read_args(target), "call_read")
    partition.append(ContextPartitionItem(
        call_id="call_read",
        name="file_tool",
        arguments=_read_args(target),
        type="function_call",
    ))
    partition.append(read_result)

    target.unlink()
    missing_result = await tool.call(_read_args(target), "call_missing")

    tombstoned = partition.items[1]
    assert missing_result.output.startswith("ERROR FileNotFoundError")
    assert tombstoned.prokaryotes_annotations["file_tool.status"] == "stale"
    assert "no longer accessible" in tombstoned.output
    assert "FileNotFoundError" in tombstoned.output


@pytest.mark.asyncio
async def test_failed_write_tombstones_prior_live_windows_for_same_path(tmp_path: Path):
    target = tmp_path / "write_missing.txt"
    target.write_text("gone soon\n", encoding="utf-8")
    partition = _empty_partition()
    tool = FileTool(partition, workspace_root=tmp_path)

    read_result = await tool.call(_read_args(target), "call_read")
    partition.append(ContextPartitionItem(
        call_id="call_read",
        name="file_tool",
        arguments=_read_args(target),
        type="function_call",
    ))
    partition.append(read_result)
    rev = read_result.prokaryotes_annotations["file_tool.revision"]

    target.unlink()
    write_result = await tool.call(_replace_args(target, rev, 1, 1, "replacement\n"), "call_write")

    tombstoned = partition.items[1]
    assert write_result.output.startswith("ERROR FileNotFoundError")
    assert tombstoned.prokaryotes_annotations["file_tool.status"] == "stale"
    assert "no longer accessible" in tombstoned.output
    assert "FileNotFoundError" in tombstoned.output


@pytest.mark.asyncio
async def test_write_refreshes_same_round_read_result_before_partition_append(tmp_path: Path):
    target = tmp_path / "same_round.txt"
    target.write_text("before\n", encoding="utf-8")
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

    read_result = await tool.call(_read_args(target), "call_read")
    write_result = await tool.call(
        _replace_args(target, _hash("before\n"), 1, 1, "after\n"),
        "call_write",
    )

    assert write_result.output.startswith("EDITED ")
    assert read_result.prokaryotes_annotations["file_tool.revision"] == _hash("after\n")
    assert "1 | after" in read_result.output
    assert "1 | before" not in read_result.output


@pytest.mark.asyncio
async def test_pending_result_items_are_pruned_after_partition_append(tmp_path: Path):
    target = tmp_path / "pending.txt"
    target.write_text("before\n", encoding="utf-8")
    partition = _empty_partition()
    tool = FileTool(partition, workspace_root=tmp_path)

    read_result = await tool.call(_read_args(target), "call_read")
    assert sum(item is read_result for item in tool._refreshable_items()) == 1

    partition.append(ContextPartitionItem(
        call_id="call_read",
        name="file_tool",
        arguments=_read_args(target),
        type="function_call",
    ))
    partition.append(read_result)
    refreshable_items = tool._refreshable_items()

    assert sum(item is read_result for item in refreshable_items) == 1
    assert tool._pending_result_items == []


@pytest.mark.asyncio
async def test_reconcile_tracked_files_tombstones_when_path_disappears(tmp_path: Path):
    target = tmp_path / "vanishing.txt"
    target.write_text("here today\n", encoding="utf-8")
    partition = _empty_partition()
    tool = FileTool(partition, workspace_root=tmp_path)

    read_result = await tool.call(_read_args(target), "call_t")
    partition.append(ContextPartitionItem(
        call_id="call_t",
        name="file_tool",
        arguments=_read_args(target),
        type="function_call",
    ))
    partition.append(read_result)

    target.unlink()
    await reconcile_tracked_files(partition, workspace_root=tmp_path)

    tombstoned = partition.items[1]
    assert tombstoned.prokaryotes_annotations["file_tool.status"] == "stale"
    assert "no longer accessible" in tombstoned.output
    assert "FileNotFoundError" in tombstoned.output


@pytest.mark.asyncio
async def test_reconcile_tracked_files_tombstones_when_path_now_escapes_workspace(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    target = workspace / "tracked.txt"
    outside = outside_dir / "secret.txt"
    target.write_text("inside\n", encoding="utf-8")
    outside.write_text("outside\n", encoding="utf-8")
    partition = _empty_partition()
    tool = FileTool(partition, workspace_root=workspace)

    read_result = await tool.call(_read_args(target), "call_escape")
    partition.append(ContextPartitionItem(
        call_id="call_escape",
        name="file_tool",
        arguments=_read_args(target),
        type="function_call",
    ))
    partition.append(read_result)

    target.unlink()
    target.symlink_to(outside)
    await reconcile_tracked_files(partition, workspace_root=workspace)

    tombstoned = partition.items[1]
    assert tombstoned.prokaryotes_annotations["file_tool.status"] == "stale"
    assert "no longer accessible" in tombstoned.output
    assert "ValueError" in tombstoned.output
    assert "outside" not in tombstoned.output


@pytest.mark.asyncio
async def test_reconcile_tracked_files_skips_items_already_at_current_revision(tmp_path: Path):
    target = tmp_path / "stable.txt"
    target.write_text("unchanged\n", encoding="utf-8")
    partition = _empty_partition()
    tool = FileTool(partition, workspace_root=tmp_path)

    read_result = await tool.call(_read_args(target), "call_s")
    partition.append(ContextPartitionItem(
        call_id="call_s",
        name="file_tool",
        arguments=_read_args(target),
        type="function_call",
    ))
    partition.append(read_result)
    output_before = read_result.output

    await reconcile_tracked_files(partition, workspace_root=tmp_path)

    assert partition.items[1].output == output_before


def test_refresh_live_windows_handles_multiple_views_into_same_path():
    text_v1 = "a\nb\nc\nd\ne\n"
    text_v2 = "A\nB\nC\nD\nE\n"
    rev_v1 = _hash(text_v1)
    rev_v2 = _hash(text_v2)
    item_first = ContextPartitionItem(
        call_id="c1",
        type="function_call_output",
        output="placeholder",
        prokaryotes_annotations={
            "file_tool.path": "/tmp/x",
            "file_tool.revision": rev_v1,
            "file_tool.status": "live",
            "file_tool.view_start_line": "1",
            "file_tool.view_end_line": "3",
        },
    )
    item_second = ContextPartitionItem(
        call_id="c2",
        type="function_call_output",
        output="placeholder",
        prokaryotes_annotations={
            "file_tool.path": "/tmp/x",
            "file_tool.revision": rev_v1,
            "file_tool.status": "live",
            "file_tool.view_start_line": "3",
            "file_tool.view_end_line": "5",
        },
    )

    _refresh_live_windows([item_first, item_second], "/tmp/x", text_v2, rev_v2)

    assert item_first.prokaryotes_annotations["file_tool.revision"] == rev_v2
    assert "1 | A" in item_first.output
    assert item_second.prokaryotes_annotations["file_tool.revision"] == rev_v2
    assert "3 | C" in item_second.output


def test_render_view_returns_empty_view_past_eof():
    end_line, line_count, view_lines = render_view("a\nb\n", start_line=10, max_lines=5)

    assert line_count == 2
    assert view_lines == []
    assert end_line == 9


@pytest.mark.asyncio
async def test_concurrent_writes_same_path_yield_one_edit_one_conflict(tmp_path: Path):
    target = tmp_path / "shared.txt"
    target.write_text("x\n", encoding="utf-8")
    rev_a = _hash("x\n")
    tool_a = FileTool(_empty_partition(), workspace_root=tmp_path)
    tool_b = FileTool(_empty_partition(), workspace_root=tmp_path)

    result_a, result_b = await asyncio.gather(
        tool_a.call(_replace_args(target, rev_a, 1, 1, "A\n"), "call_a"),
        tool_b.call(_replace_args(target, rev_a, 1, 1, "B\n"), "call_b"),
    )

    outputs = (result_a.output, result_b.output)
    assert sum(o.startswith("EDITED ") for o in outputs) == 1
    assert sum(o.startswith("CONFLICT ") for o in outputs) == 1
    final = target.read_text(encoding="utf-8")
    assert final in {"A\n", "B\n"}
    conflict_result = result_a if result_a.output.startswith("CONFLICT ") else result_b
    assert conflict_result.prokaryotes_annotations["file_tool.revision"] == _hash(final)


@pytest.mark.asyncio
async def test_concurrent_writes_different_paths_do_not_block_each_other(tmp_path: Path):
    target_a = tmp_path / "a.txt"
    target_b = tmp_path / "b.txt"
    target_a.write_text("a\n", encoding="utf-8")
    target_b.write_text("b\n", encoding="utf-8")
    tool_a = FileTool(_empty_partition(), workspace_root=tmp_path)
    tool_b = FileTool(_empty_partition(), workspace_root=tmp_path)

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
    lock_one_a = FileTool._get_path_lock(path_one)
    lock_one_b = FileTool._get_path_lock(path_one)
    lock_two = FileTool._get_path_lock(path_two)

    assert lock_one_a is lock_one_b
    assert lock_two is not lock_one_a


@pytest.mark.asyncio
async def test_conflict_refreshes_prior_live_windows_for_same_path(tmp_path: Path):
    target = tmp_path / "drift.txt"
    target.write_text("old1\nold2\n", encoding="utf-8")
    partition = _empty_partition()
    tool = FileTool(partition, workspace_root=tmp_path)

    read_result = await tool.call(_read_args(target), "call_read")
    partition.append(ContextPartitionItem(
        call_id="call_read",
        name="file_tool",
        arguments=_read_args(target),
        type="function_call",
    ))
    partition.append(read_result)
    rev_a = read_result.prokaryotes_annotations["file_tool.revision"]

    target.write_text("new1\nnew2\nnew3\n", encoding="utf-8")
    rev_b = _hash("new1\nnew2\nnew3\n")

    write_result = await tool.call(
        _replace_args(target, rev_a, 1, 1, "X\n"),
        "call_write",
    )

    assert write_result.output.startswith("CONFLICT ")
    assert write_result.prokaryotes_annotations["file_tool.revision"] == rev_b
    refreshed = partition.items[1]
    assert refreshed.prokaryotes_annotations["file_tool.revision"] == rev_b
    assert refreshed.prokaryotes_annotations["file_tool.status"] == "live"
    assert "1 | new1" in refreshed.output
    assert "3 | new3" in refreshed.output


@pytest.mark.asyncio
async def test_flock_alone_serializes_concurrent_locked_write_transactions(tmp_path: Path):
    target = tmp_path / "flock_target.txt"
    target.write_text("v0\n", encoding="utf-8")
    rev = _hash("v0\n")
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)

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
    tool = FileTool(_empty_partition(), workspace_root=tmp_path)
    path_lock = FileTool._get_path_lock(str(target.resolve()))

    async with path_lock:
        target.write_text("partial\n", encoding="utf-8")
        read_task = asyncio.create_task(tool.call(_read_args(target), "call_read"))
        await asyncio.sleep(0)
        assert not read_task.done()
        target.write_text("final\n", encoding="utf-8")

    result = await asyncio.wait_for(read_task, timeout=5)
    assert result.prokaryotes_annotations["file_tool.revision"] == _hash("final\n")
    assert "1 | final" in result.output
    assert "partial" not in result.output


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

    read_task = asyncio.create_task(asyncio.to_thread(_locked_read_text, target))
    await asyncio.sleep(0.05)
    assert not read_task.done()
    release_writer.set()

    text = await asyncio.wait_for(read_task, timeout=5)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert text == "final\n"
