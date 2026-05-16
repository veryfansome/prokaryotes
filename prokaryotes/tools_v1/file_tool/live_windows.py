from prokaryotes.api_v1.models import (
    ContextPartition,
    ContextPartitionItem,
    conversation_message_items,
)
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


def _has_transient_file_diagnostic(item: ContextPartitionItem) -> bool:
    """Return True if `item`'s output carries a diagnostic header that should re-render on
    the next refresh even at the same revision, so the header gets dropped once the
    diagnostic's condition no longer applies."""
    output = item.output or ""
    return (
        output.startswith("ALREADY_EXISTS ")
        or output.startswith("CONFLICT ")
        or output.startswith("RANGE_ERROR ")
        or output.startswith("RANGE_TRUNCATED ")
    )


def _is_unstable_coverage(item: ContextPartitionItem) -> bool:
    """Return True if `item` should not be trusted as coverage for redundant-read detection.

    ALREADY_EXISTS, CONFLICT, and RANGE_ERROR carry decision-relevant state the model still
    needs to react to before consuming the embedded view. RANGE_TRUNCATED is intentionally
    excluded: its diagnostic header is about the request being over-cap, not about file state,
    and the embedded view is a real live window over [view_start_line, view_end_line] with
    `requested_end_line` set to the cap — i.e. stable coverage for its returned span."""
    output = item.output or ""
    return output.startswith("ALREADY_EXISTS ") or output.startswith("CONFLICT ") or output.startswith("RANGE_ERROR ")


def _live_window_stable_repr(item: ContextPartitionItem) -> tuple:
    """Identity tuple for a live-window item — only the fields that don't mutate
    during `_refresh_live_windows`. The mutable trio (`output`,
    `file_tool.revision`, `file_tool.view_end_line`) is intentionally excluded."""
    ann = item.prokaryotes_annotations or {}
    return (
        item.type,
        item.call_id,
        item.id,
        ann.get("file_tool.path"),
        ann.get("file_tool.status"),
        ann.get("file_tool.view_start_line"),
        ann.get("file_tool.requested_end_line"),
    )


def _message_count_before_item_index(items: list[ContextPartitionItem], item_index: int) -> int:
    return len(conversation_message_items(items[:item_index]))


def _refresh_live_windows(
    items: list[ContextPartitionItem],
    path: str,
    text: str,
    revision: str,
    max_lines: int,
) -> int:
    """Re-render every live window for `path` against `text` / `revision`. Items already
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
            text,
            start_line,
            max_lines,
            requested_end_line=requested_end_line,
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


def _strip_live_window_output(item: ContextPartitionItem) -> str:
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
        diagnostic_lines = []
        for line in output.splitlines():
            if line.startswith(CURRENT_VIEW_MARKER_PREFIX):
                if diagnostic_lines:
                    return "\n".join(diagnostic_lines) + "\n" + placeholder
                break
            diagnostic_lines.append(line)
    return placeholder


def _tombstone_live_windows(
    items: list[ContextPartitionItem],
    path: str,
    reason: str,
) -> None:
    tombstone = render_tombstone(path, reason)
    for item in items:
        ann = item.prokaryotes_annotations or {}
        if ann.get("file_tool.path") == path and ann.get("file_tool.status") == "live":
            item.prokaryotes_annotations["file_tool.status"] = "stale"
            item.output = tombstone


def _tool_round_start_index(items: list[ContextPartitionItem], idx: int) -> int:
    """Return the earliest index we can insert at without splitting a tool-call round.

    Provider histories represent one parallel tool-call round as a contiguous block of
    `function_call` items followed by a contiguous block of `function_call_output` items.
    When the first active file item is inside that structure, insertion must happen before
    the whole block, otherwise Anthropic can see a tool_use whose matching tool_result is
    delayed past an unrelated lifted pair.
    """
    start = idx
    while start > 0 and items[start - 1].type == "function_call_output":
        start -= 1
    while start > 0 and items[start - 1].type == "function_call":
        start -= 1
    return start


def is_live_window(item: ContextPartitionItem) -> bool:
    """A live-window function_call_output is the one mutable kind of partition item:
    its `output` and the `file_tool.revision` / `file_tool.view_end_line` annotations
    can change in-place when `reconcile_tracked_files` or a `file_tool` write refreshes
    it. Stable live-window annotations such as `file_tool.view_start_line` and
    `file_tool.requested_end_line` remain part of the item's identity. All other item
    kinds (messages, function_calls, edit records, tombstoned outputs) are append-only
    and compared with full Pydantic equality."""
    ann = item.prokaryotes_annotations
    return item.type == "function_call_output" and ann is not None and ann.get("file_tool.status") == "live"


def items_equal_mod_live_windows(
    a: list[ContextPartitionItem],
    b: list[ContextPartitionItem],
) -> bool:
    """Compare two item lists treating live-window refresh as equivalent.

    Used by `_compact_partition`'s prefix check so that a concurrent request that ran
    `reconcile_tracked_files` or applied `file_tool` writes between the snapshot and
    the swap doesn't falsify the prefix purely by refreshing earlier live windows
    in-place. Live -> stale tombstoning is still surfaced as a difference because
    `file_tool.status` is part of the stable identity tuple. Real divergences
    (appended items, replaced items, modified message content, edit-record changes)
    fall through to full equality and continue to skip the swap.

    The lifted windows in the resulting swapped partition come from `snapshot.items`
    and may briefly carry pre-refresh `output`/`revision` until the next request's
    `reconcile_tracked_files` repairs them — a one-turn lag rather than a regression.
    """
    if len(a) != len(b):
        return False
    for ai, bi in zip(a, b, strict=True):
        if is_live_window(ai) and is_live_window(bi):
            if _live_window_stable_repr(ai) != _live_window_stable_repr(bi):
                return False
        elif ai != bi:
            return False
    return True


def lift_active_live_windows(
    pre_tail: list[ContextPartitionItem],
    recency_tail: list[ContextPartitionItem],
) -> list[ContextPartitionItem]:
    """Lift pre-tail live windows for paths active in the recency tail.

    Active paths are those carrying a `file_tool.path` annotation in the recency tail
    (live windows or edit records). For each active path, every pre-tail
    `(function_call, function_call_output)` pair whose output is a live window for that
    path is moved into the new tail immediately before the tool-call round that
    first carries a `file_tool.path` annotation. Original `call_id`s and arguments
    are preserved.

    The placement is constrained by Anthropic's user-first message requirement and the
    fact that `recency_tail_items()` guarantees the tail's leading message is user role:
    inserting before the annotated tool round slots lifted pairs after that leading user
    prefix while keeping them adjacent to the downstream activity that uses them without
    splitting same-round tool calls from their outputs.
    """
    tail_function_call_idx_by_call_id: dict[str, int] = {}
    for idx, item in enumerate(recency_tail):
        if item.type == "function_call":
            cid = item.call_id or item.id
            if cid is not None:
                tail_function_call_idx_by_call_id[cid] = idx

    active_paths: set[str] = set()
    insertion_idx: int | None = None
    for idx, item in enumerate(recency_tail):
        ann = item.prokaryotes_annotations or {}
        path = ann.get("file_tool.path")
        if not path:
            continue
        if ann.get("file_tool.status") == "stale":
            continue
        active_paths.add(path)
        if insertion_idx is not None:
            continue
        if item.type == "function_call_output":
            cid = item.call_id or item.id
            paired_call_idx = tail_function_call_idx_by_call_id.get(cid) if cid else None
            if paired_call_idx is not None:
                insertion_idx = _tool_round_start_index(recency_tail, paired_call_idx)
            else:
                insertion_idx = _tool_round_start_index(recency_tail, idx)
        else:
            insertion_idx = _tool_round_start_index(recency_tail, idx)
    if not active_paths:
        return list(recency_tail)

    pre_tail_function_calls: dict[str, ContextPartitionItem] = {}
    for item in pre_tail:
        if item.type == "function_call":
            cid = item.call_id or item.id
            if cid is not None:
                pre_tail_function_calls[cid] = item

    lifted_pairs: list[ContextPartitionItem] = []
    for item in pre_tail:
        if item.type != "function_call_output":
            continue
        ann = item.prokaryotes_annotations or {}
        if ann.get("file_tool.status") != "live":
            continue
        if ann.get("file_tool.path") not in active_paths:
            continue
        cid = item.call_id or item.id
        if cid is None:
            continue
        function_call_item = pre_tail_function_calls.get(cid)
        if function_call_item is None:
            continue
        lifted_pairs.append(function_call_item)
        lifted_pairs.append(item)

    if not lifted_pairs:
        return list(recency_tail)

    insert_at = insertion_idx if insertion_idx is not None else 0
    return list(recency_tail[:insert_at]) + lifted_pairs + list(recency_tail[insert_at:])


def recency_tail_items(
    items: list[ContextPartitionItem],
    message_tail_count: int,
) -> tuple[list[ContextPartitionItem], int]:
    message_indexes = [
        idx for idx, item in enumerate(items) if item.type == "message" and item.role in {"user", "assistant"}
    ]
    if not message_indexes:
        return [], 0
    first_tail_message_pos = max(0, len(message_indexes) - message_tail_count)
    while (
        first_tail_message_pos < len(message_indexes) and items[message_indexes[first_tail_message_pos]].role != "user"
    ):
        first_tail_message_pos += 1
    if first_tail_message_pos >= len(message_indexes):
        return [], 0
    first_tail_item_index = message_indexes[first_tail_message_pos]
    return items[first_tail_item_index:], _message_count_before_item_index(items, first_tail_item_index)


def strip_live_window_bodies(partition: ContextPartition) -> ContextPartition:
    """Return a deep copy of `partition` with each live-window `output` rewritten so the
    summarizer cannot fossilize current file contents into `ancestor_summaries`.

    The invariant this enforces is broad: *no* live window's current file contents reach
    the summary input — not just live windows for paths active in the recency tail. Once
    a summary is written, no later `reconcile_tracked_files` can reach back into it, so
    any file body that lands there drifts out of sync with the on-disk truth that future
    live-window refreshes track. Stripping at the summarization input is the only place
    we can prevent that.

    Two stripping shapes, depending on the live-window kind (detected by `output` prefix):

    - Ordinary `FILE ... status=live` `read_lines` results are replaced wholesale with a
      path-only placeholder. The diagnostic value is just the path.
    - `ALREADY_EXISTS`, `CONFLICT`, `RANGE_ERROR`, and `RANGE_TRUNCATED` results keep
      their header lines (which describe what the model tried and the failure or
      truncation mode) and have only the embedded `Current view ...` body — the actual
      file contents — replaced with the placeholder. That preserves the historical
      diagnostic signal in the summary while keeping current content out.

    Edit records, tombstones (`status=stale`), function_call items, and message items
    are not touched: edit records and tombstones already document file activity without
    embedding current contents, and the rest are not file-related.
    """
    stripped = partition.model_copy(deep=True)
    for item in stripped.items:
        if not is_live_window(item):
            continue
        item.output = _strip_live_window_output(item)
    return stripped
