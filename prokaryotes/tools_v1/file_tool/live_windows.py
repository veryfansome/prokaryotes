"""Working-file window helpers: refresh, tombstone, normalize.

Working files are first-class state on `Conversation.working_file_windows`; the compactor blanks them on its
summarization-input snapshot rather than rewriting historical TurnItems.

These helpers mutate a `list[WorkingFileWindow]` in place — typically `conversation.working_file_windows`.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from prokaryotes.conversation_v1.models import WorkingFileWindow
from prokaryotes.tools_v1.file_tool.paths import (
    FileToolFileTooLargeError,
    _resolve_path,
)
from prokaryotes.tools_v1.file_tool.reads import _read_text_under_file_tool_lock
from prokaryotes.tools_v1.file_tool.rendering import (
    render_live_window,
    render_tombstone,
    render_view,
)

_DIAGNOSTIC_SOURCE_KINDS = frozenset({"range_truncated", "already_exists", "conflict", "range_error"})


def refresh_windows_for_path(
    windows: list[WorkingFileWindow],
    path: str,
    text: str,
    revision: str,
    max_lines: int,
    exclude_window_ids: set[str] | None = None,
) -> int:
    """Re-render every live window for `path` against `text`/`revision`. Windows already at `revision` with a
    non-diagnostic `source_kind` are left alone. Diagnostic windows normalize back to `source_kind='read_lines'`.

    `exclude_window_ids` skips refresh for windows just minted in the current call — `FileTool._do_write` and
    `FileTool._do_create_file` pass the call_id so the diagnostic window they just appended doesn't get
    normalized in the same turn.

    Returns the number of windows rewritten.
    """
    excluded = exclude_window_ids or set()
    refreshed = 0
    for window in windows:
        if window.path != path or window.status != "live":
            continue
        if window.window_id in excluded:
            continue
        if window.revision == revision and window.source_kind not in _DIAGNOSTIC_SOURCE_KINDS:
            continue
        end_line, _line_count, view_lines = render_view(
            text, window.view_start_line, max_lines, requested_end_line=window.requested_end_line
        )
        window.rendered_output = render_live_window(
            path=path,
            revision=revision,
            start_line=window.view_start_line,
            end_line=end_line,
            line_count=_line_count,
            view_lines=view_lines,
        )
        window.revision = revision
        window.view_end_line = end_line
        if window.source_kind in _DIAGNOSTIC_SOURCE_KINDS:
            window.source_kind = "read_lines"
        refreshed += 1
    return refreshed


def tombstone_windows_for_path(windows: list[WorkingFileWindow], path: str, reason: str) -> None:
    """Mark every window for `path` as `status='stale'` with `source_kind='tombstone'`; rewrite `rendered_output`
    to the tombstone marker.
    """
    rendered = render_tombstone(path, reason)
    for window in windows:
        if window.path != path:
            continue
        window.status = "stale"
        window.source_kind = "tombstone"
        window.rendered_output = rendered


async def reconcile_working_files(
    windows: list[WorkingFileWindow],
    workspace_root: Path,
    *,
    max_file_bytes: int,
    max_lines: int,
) -> None:
    """Refresh every distinct live path against on-disk state, in place.

    For each unique `(path)` among live windows, read the current file revision and refresh all matching windows.
    Failures (file gone, too large, permission denied, etc.) tombstone every window for that path.
    """
    live_paths = {w.path for w in windows if w.status == "live"}
    for path_str in sorted(live_paths):
        try:
            path = _resolve_path(path_str, workspace_root)
            current_text = await _read_text_under_file_tool_lock(path, max_file_bytes)
        except (
            FileNotFoundError,
            FileToolFileTooLargeError,
            IsADirectoryError,
            PermissionError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:
            tombstone_windows_for_path(windows, path_str, type(exc).__name__)
            continue
        current_revision = sha256(current_text.encode("utf-8")).hexdigest()
        refresh_windows_for_path(windows, path_str, current_text, current_revision, max_lines)
