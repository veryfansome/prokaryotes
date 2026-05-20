"""`reconcile_working_files` — refresh live windows, normalize diagnostic source_kinds, tombstone on access failure."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from prokaryotes.conversation_v1.models import WorkingFileWindow
from prokaryotes.tools_v1.file_tool.live_windows import reconcile_working_files


def _window(
    window_id: str,
    path: str,
    *,
    revision: str,
    source_kind: str = "read_lines",
) -> WorkingFileWindow:
    return WorkingFileWindow(
        window_id=window_id,
        path=path,
        status="live",
        revision=revision,
        rendered_output=f"FILE path={path} revision={revision} status=live",
        view_start_line=1,
        view_end_line=3,
        requested_end_line=3,
        source_kind=source_kind,
    )


@pytest.mark.asyncio
async def test_refreshes_window_when_disk_revision_changed(tmp_path: Path):
    target = tmp_path / "a.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    new_revision = sha256(target.read_bytes()).hexdigest()
    windows = [_window("c-1", str(target), revision="stale-revision")]
    await reconcile_working_files(
        windows,
        workspace_root=tmp_path,
        max_file_bytes=1_000_000,
        max_lines=200,
    )
    assert windows[0].revision == new_revision


@pytest.mark.asyncio
async def test_normalizes_diagnostic_source_kind_to_read_lines(tmp_path: Path):
    target = tmp_path / "a.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    windows = [_window("c-1", str(target), revision="stale-revision", source_kind="conflict")]
    await reconcile_working_files(
        windows,
        workspace_root=tmp_path,
        max_file_bytes=1_000_000,
        max_lines=200,
    )
    assert windows[0].source_kind == "read_lines"


@pytest.mark.asyncio
async def test_tombstones_when_path_is_gone(tmp_path: Path):
    absent = tmp_path / "absent.txt"
    windows = [_window("c-1", str(absent), revision="r1")]
    await reconcile_working_files(
        windows,
        workspace_root=tmp_path,
        max_file_bytes=1_000_000,
        max_lines=200,
    )
    assert windows[0].status == "stale"
    assert windows[0].source_kind == "tombstone"


@pytest.mark.asyncio
async def test_already_at_current_revision_is_left_alone(tmp_path: Path):
    target = tmp_path / "a.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    revision = sha256(target.read_bytes()).hexdigest()
    original = "FILE path=... revision=already-current\nVIEW BODY"
    window = WorkingFileWindow(
        window_id="c-1",
        path=str(target),
        status="live",
        revision=revision,
        rendered_output=original,
        view_start_line=1,
        view_end_line=3,
        requested_end_line=3,
        source_kind="read_lines",
    )
    windows = [window]
    await reconcile_working_files(
        windows,
        workspace_root=tmp_path,
        max_file_bytes=1_000_000,
        max_lines=200,
    )
    # Output preserved verbatim
    assert windows[0].rendered_output == original
