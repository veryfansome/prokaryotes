import asyncio
import json
from hashlib import sha256
from pathlib import Path

import pytest

from prokaryotes.api_v1.models import ContextPartition, ContextPartitionItem
from prokaryotes.tools_v1.file_tool import (
    FileTool,
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
    # Edit record carries only file_tool.path — never refreshed, used by compaction lift.
    assert write_result.prokaryotes_annotations == {"file_tool.path": str(target.resolve())}
    assert write_result.output.startswith("EDITED ")
    assert "Removed (lines 2-3):" in write_result.output
    assert "Added (lines 2-3):" in write_result.output
    assert "line_count: 4 → 4" in write_result.output

    # The earlier read window has been refreshed in-place to reflect the new content.
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
    # Conflict result doubles as a fresh live window so reconciliation can refresh it later
    # and the model can immediately retry against current_revision.
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
    # File is unchanged.
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

    # External edit between turns.
    target.write_text("v1\nv2\nv3\n", encoding="utf-8")

    await reconcile_tracked_files(partition)

    refreshed = partition.items[1]
    assert refreshed.prokaryotes_annotations["file_tool.revision"] == _hash("v1\nv2\nv3\n")
    assert "3 | v3" in refreshed.output
    assert refreshed.prokaryotes_annotations["file_tool.status"] == "live"


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
    await reconcile_tracked_files(partition)

    tombstoned = partition.items[1]
    assert tombstoned.prokaryotes_annotations["file_tool.status"] == "stale"
    assert "no longer accessible" in tombstoned.output
    assert "FileNotFoundError" in tombstoned.output


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

    await reconcile_tracked_files(partition)

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
    # end_line marker indicates an empty view.
    assert end_line == 9


@pytest.mark.asyncio
async def test_concurrent_writes_same_path_yield_one_edit_one_conflict(tmp_path: Path):
    """B1 regression: two FileTool instances writing the same path with the same
    expected_revision must serialize so that exactly one applies and the other reports a
    conflict. Without the class-level per-path lock both would pass the revision check on
    revision A and silently overwrite each other."""
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
    # The conflicting result carries a fresh live-window view of whichever write won.
    conflict_result = result_a if result_a.output.startswith("CONFLICT ") else result_b
    assert conflict_result.prokaryotes_annotations["file_tool.revision"] == _hash(final)


@pytest.mark.asyncio
async def test_concurrent_writes_different_paths_do_not_block_each_other(tmp_path: Path):
    """Different resolved paths get different locks, so unrelated writes must proceed
    in parallel without serializing on each other."""
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
    """The class-level path-lock map must dedupe by path so two FileTool instances
    coordinate on the same Lock object even though they don't share `self._lock`."""
    path_one = str(tmp_path / "one.txt")
    path_two = str(tmp_path / "two.txt")
    lock_one_a = FileTool._get_path_lock(path_one)
    lock_one_b = FileTool._get_path_lock(path_one)
    lock_two = FileTool._get_path_lock(path_two)

    assert lock_one_a is lock_one_b
    assert lock_two is not lock_one_a


@pytest.mark.asyncio
async def test_conflict_refreshes_prior_live_windows_for_same_path(tmp_path: Path):
    """Codex P1 regression: a stale-revision write must refresh older live windows for
    the same path to the just-discovered current revision, not leave them at the stale
    revision while the conflict result alone reports the new state."""
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

    # External edit between the read and the conflicting write.
    target.write_text("new1\nnew2\nnew3\n", encoding="utf-8")
    rev_b = _hash("new1\nnew2\nnew3\n")

    write_result = await tool.call(
        _replace_args(target, rev_a, 1, 1, "X\n"),
        "call_write",
    )

    assert write_result.output.startswith("CONFLICT ")
    assert write_result.prokaryotes_annotations["file_tool.revision"] == rev_b
    # The earlier read window in the partition must have been refreshed in-place.
    refreshed = partition.items[1]
    assert refreshed.prokaryotes_annotations["file_tool.revision"] == rev_b
    assert refreshed.prokaryotes_annotations["file_tool.status"] == "live"
    assert "1 | new1" in refreshed.output
    assert "3 | new3" in refreshed.output


@pytest.mark.asyncio
async def test_flock_alone_serializes_concurrent_locked_write_transactions(tmp_path: Path):
    """Codex P2 regression: drive `_locked_write_transaction` concurrently in two threads
    while bypassing `_do_write` (and therefore the per-path `asyncio.Lock`). The only thing
    that can serialize the read-check-write critical sections is `fcntl.flock(LOCK_EX)` on
    the target file's descriptor. Without flock both threads would pass `expected_revision`
    against the same starting state and one write would silently overwrite the other."""
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
