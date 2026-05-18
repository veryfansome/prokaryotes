"""Live-window helpers retyped for `TurnItem`.

The lift-related helpers (`lift_active_live_windows`, `_tool_round_start_index`,
`recency_tail_items`, `_message_count_before_item_index`,
`items_equal_mod_live_windows`) are removed — that selection now lives in
`ConversationCompactor._compute_lift_plan` and operates on `TurnExecution`s
rather than a flat item list. The new compactor's CAS prefix check
(`_messages_match_prefix`) operates on `Conversation.messages`, which are
append-only with tombstone flags — no `mod_live_windows` carve-out needed.
"""

from __future__ import annotations

from prokaryotes.conversation_v1.models import TurnItem
from prokaryotes.tools_v1.file_tool.rendering import (
    CURRENT_VIEW_MARKER_PREFIX,
    render_live_window,
    render_tombstone,
    render_view,
)


def _annotation_requested_end_line(annotations: dict[str, str]) -> int | None:
    requested_end_line = annotations.get("file_tool.requested_end_line")
    if requested_end_line is None:
        return None
    try:
        return int(requested_end_line)
    except ValueError:
        return None


def _has_transient_file_diagnostic(item: TurnItem) -> bool:
    output = item.output or ""
    return (
        output.startswith("ALREADY_EXISTS ")
        or output.startswith("CONFLICT ")
        or output.startswith("RANGE_ERROR ")
        or output.startswith("RANGE_TRUNCATED ")
    )


def _is_unstable_coverage(item: TurnItem) -> bool:
    """`ALREADY_EXISTS`, `CONFLICT`, `RANGE_ERROR` carry decision-relevant state the model
    still needs to react to before consuming the embedded view. `RANGE_TRUNCATED` is
    intentionally excluded — its embedded view is real coverage for the returned span."""
    output = item.output or ""
    return output.startswith("ALREADY_EXISTS ") or output.startswith("CONFLICT ") or output.startswith("RANGE_ERROR ")


def _refresh_live_windows(
    items: list[TurnItem],
    path: str,
    text: str,
    revision: str,
    max_lines: int,
) -> int:
    """Re-render every live window for `path` against `text`/`revision`. Items already
    at `revision` are left alone. Returns the number of items actually rewritten."""
    refreshed_count = 0
    for item in items:
        ann = item.prokaryotes_annotations
        if not ann or ann.get("file_tool.path") != path or ann.get("file_tool.status") != "live":
            continue
        if ann.get("file_tool.revision") == revision and not _has_transient_file_diagnostic(item):
            continue
        try:
            start_line = int(ann["file_tool.view_start_line"])
        except (KeyError, ValueError):
            continue
        requested_end_line = _annotation_requested_end_line(ann)
        end_line, line_count, view_lines = render_view(
            text, start_line, max_lines, requested_end_line=requested_end_line
        )
        item.output = render_live_window(
            path=path,
            revision=revision,
            start_line=start_line,
            end_line=end_line,
            line_count=line_count,
            view_lines=view_lines,
        )
        ann["file_tool.revision"] = revision
        ann["file_tool.view_end_line"] = str(end_line)
        refreshed_count += 1
    return refreshed_count


def _strip_live_window_output(item: TurnItem) -> str:
    ann = item.prokaryotes_annotations or {}
    path = ann.get("file_tool.path", "<unknown>")
    placeholder = (
        f"[Live tracked file: {path} — current contents are tracked via the live-window"
        f" mechanism on subsequent turns, not summarized here.]"
    )
    output = item.output or ""
    if (
        output.startswith("ALREADY_EXISTS ")
        or output.startswith("CONFLICT ")
        or output.startswith("RANGE_ERROR ")
        or output.startswith("RANGE_TRUNCATED ")
    ):
        diagnostic_lines: list[str] = []
        for line in output.splitlines():
            if line.startswith(CURRENT_VIEW_MARKER_PREFIX):
                if diagnostic_lines:
                    return "\n".join(diagnostic_lines) + "\n" + placeholder
                break
            diagnostic_lines.append(line)
    return placeholder


def _tombstone_live_windows(items: list[TurnItem], path: str, reason: str) -> None:
    tombstone = render_tombstone(path, reason)
    for item in items:
        ann = item.prokaryotes_annotations or {}
        if ann.get("file_tool.path") == path and ann.get("file_tool.status") == "live":
            item.prokaryotes_annotations["file_tool.status"] = "stale"
            item.output = tombstone


def is_live_window(item: TurnItem) -> bool:
    ann = item.prokaryotes_annotations
    return item.type == "function_call_output" and ann is not None and ann.get("file_tool.status") == "live"


def strip_live_window_bodies(items: list[TurnItem]) -> list[TurnItem]:
    """Return a deep copy of `items` with each live-window `output` rewritten so the
    summarizer cannot fossilize current file contents into `ancestor_summaries`.

    The compactor's summarization pass calls this on pre-tail TurnExecutions before
    projecting them for the summary LLM call. The invariant is broad: no live
    window's current file contents reach the summary input.
    """
    stripped: list[TurnItem] = []
    for item in items:
        copy_item = item.model_copy(deep=True)
        if is_live_window(copy_item):
            copy_item.output = _strip_live_window_output(copy_item)
        stripped.append(copy_item)
    return stripped
