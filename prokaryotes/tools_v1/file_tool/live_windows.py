"""Working-file window helpers: refresh, tombstone, normalize.

Working files are first-class state on `Conversation.working_file_windows`; the compactor blanks them on its
summarization-input snapshot rather than rewriting historical TurnItems.

These helpers mutate a `list[WorkingFileWindow]` in place — typically `conversation.working_file_windows`.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from uuid import uuid4

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

DIAGNOSTIC_SOURCE_KINDS = frozenset({"already_exists", "conflict", "range_error"})


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
    Each re-rendered window's `line_count` is updated to the freshly-read count. Empty-view past-EOF placeholders
    that are no longer past EOF (`view_start_line <= line_count` post-refresh) are retired.

    `exclude_window_ids` skips refresh for windows just minted in the current call — `FileTool._do_write` and
    `FileTool._do_create_file` pass the call_id so the diagnostic window they just appended doesn't get
    normalized in the same turn.

    Returns the number of windows rewritten.
    """
    excluded = exclude_window_ids or set()
    refreshed = 0
    obsolete_placeholder_ids: list[str] = []
    for window in windows:
        if window.path != path or window.status != "live":
            continue
        if window.window_id in excluded:
            continue
        if window.revision == revision and window.source_kind not in DIAGNOSTIC_SOURCE_KINDS:
            # Unchanged at this revision and not a diagnostic — content and line_count are already current, nothing
            # to re-render. No obsolete-placeholder check is needed here: a placeholder is always minted with
            # view_start_line > line_count, so its position can only gain content when the file grows — a revision
            # change that routes through the re-render branch below, whose post-render check retires it in that same
            # pass. By the time a refresh sees this window already at the current revision, an obsoleted placeholder
            # is gone.
            continue
        end_line, line_count, view_lines = render_view(
            text, window.view_start_line, max_lines, requested_end_line=window.requested_end_line
        )
        window.rendered_output = render_live_window(
            path=path,
            revision=revision,
            start_line=window.view_start_line,
            end_line=end_line,
            line_count=line_count,
            view_lines=view_lines,
        )
        window.revision = revision
        window.view_end_line = end_line
        window.line_count = line_count
        if window.source_kind in DIAGNOSTIC_SOURCE_KINDS:
            window.source_kind = "read_lines"
        refreshed += 1
        # POST-render geometry, NOT the pre-render is_empty_view: a genuine past-EOF placeholder re-renders empty
        # (end_line < view_start_line, because requested_end_line == view_start_line - 1) and is retired once the
        # file has content at its start; a content window that transiently shrank below its start (file got short)
        # re-renders back to real content here (end_line >= view_start_line) and must be kept. Using the stale
        # is_empty_view would wrongly drop the regrown content window.
        if end_line < window.view_start_line and window.view_start_line <= line_count:
            obsolete_placeholder_ids.append(window.window_id)
    if obsolete_placeholder_ids:
        stale = set(obsolete_placeholder_ids)
        windows[:] = [w for w in windows if w.window_id not in stale]
    return refreshed


def fold_windows_for_path(
    windows: list[WorkingFileWindow],
    path: str,
    text: str,
    revision: str,
    max_lines: int,
) -> None:
    """Collapse any overlap among live `read_lines`/diagnostic windows for `path` into a non-overlapping cover.

    Operates only on live, non-empty-view windows (tombstones and past-EOF placeholders are excluded). Groups
    them into maximal *overlapping* runs (contiguous-but-non-overlapping windows stay separate, preserving the
    page-through pattern). A singleton group that already fits `max_lines` passes through untouched — same
    `window_id` and `origin_call_ids`, no churn. A fused group of >=2 windows (or any group whose merged span
    exceeds `max_lines`) is retired and re-minted: merge into `[min start, max end]`, split greedily from the
    left into `max_lines`-sized chunks, and mint one `wfw-*` window per chunk with `origin_call_ids` = union of
    the group members' origins. No-op when the post-refresh state is already non-overlapping.

    Run after `refresh_windows_for_path` so source_kinds are normalized and content is current.
    """
    foldable = sorted(
        (w for w in windows if w.path == path and w.status == "live" and w.view_end_line >= w.view_start_line),
        key=lambda w: w.view_start_line,
    )
    if not foldable:
        return
    groups: list[list[WorkingFileWindow]] = []
    current = [foldable[0]]
    current_max_end = foldable[0].view_end_line
    for window in foldable[1:]:
        if window.view_start_line <= current_max_end:  # overlap (not mere adjacency)
            current.append(window)
            current_max_end = max(current_max_end, window.view_end_line)
        else:
            groups.append(current)
            current = [window]
            current_max_end = window.view_end_line
    groups.append(current)

    retire_ids: list[str] = []
    minted: list[WorkingFileWindow] = []
    for group in groups:
        span_start = min(w.view_start_line for w in group)
        span_end = max(w.view_end_line for w in group)
        if len(group) == 1 and (span_end - span_start + 1) <= max_lines:
            continue  # untouched pass-through singleton
        origins = sorted({origin for w in group for origin in w.origin_call_ids})
        retire_ids.extend(w.window_id for w in group)
        cursor = span_start
        while cursor <= span_end:
            chunk_end = min(span_end, cursor + max_lines - 1)
            end_line, line_count, view_lines = render_view(text, cursor, max_lines, requested_end_line=chunk_end)
            minted.append(
                WorkingFileWindow(
                    window_id=f"wfw-{uuid4().hex}",
                    path=path,
                    status="live",
                    revision=revision,
                    rendered_output=render_live_window(
                        path=path,
                        revision=revision,
                        start_line=cursor,
                        end_line=end_line,
                        line_count=line_count,
                        view_lines=view_lines,
                    ),
                    view_start_line=cursor,
                    view_end_line=end_line,
                    requested_end_line=chunk_end,
                    line_count=line_count,
                    origin_call_ids=origins,
                    source_kind="read_lines",
                )
            )
            cursor = chunk_end + 1
    if retire_ids:
        stale = set(retire_ids)
        windows[:] = [w for w in windows if w.window_id not in stale]
        windows.extend(minted)


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
        fold_windows_for_path(windows, path_str, current_text, current_revision, max_lines)
