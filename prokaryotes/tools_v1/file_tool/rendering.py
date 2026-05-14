CURRENT_VIEW_MARKER_PREFIX = "Current view"


def _append_live_window_refresh_note(output: str, refreshed_live_window_count: int) -> str:
    return (
        f"{output}\n\n"
        f"Live windows refreshed for this path: {refreshed_live_window_count}. "
        "Use current live windows for follow-up line numbers; this edit record is historical."
    )


def _apply_line_edit(text: str, action: str, payload: dict) -> str:
    lines = _split_into_lines(text)
    inserted = _split_into_lines(payload.get("new_text") or "")
    start = payload["start_line"]
    if action == "replace_lines":
        end = payload["end_line"]
        result = lines[: start - 1] + inserted + lines[end:]
    elif action == "insert_lines":
        result = lines[: start - 1] + inserted + lines[start - 1 :]
    elif action == "delete_lines":
        end = payload["end_line"]
        result = lines[: start - 1] + lines[end:]
    else:
        raise ValueError(f"Unsupported write action: {action!r}")
    trailing_newline = text.endswith("\n") or (text == "" and len(result) > 0)
    return "\n".join(result) + ("\n" if trailing_newline and result else "")


def _count_lines(text: str) -> int:
    return len(_split_into_lines(text))


def _render_diff_block(
    label: str,
    start_line: int,
    end_line: int,
    lines: list[str],
    max_lines: int,
) -> str:
    header = f"{label} (lines {start_line}-{end_line}):"
    if not lines:
        return header
    if len(lines) <= max_lines:
        body = "\n".join(f"{i} | {line}" for i, line in enumerate(lines, start=start_line))
        return f"{header}\n{body}"
    truncated_count = len(lines) - max_lines
    body = "\n".join(f"{i} | {line}" for i, line in enumerate(lines[:max_lines], start=start_line))
    return f"{header}\n{body}\n... {truncated_count} more lines truncated ..."


def _split_into_lines(text: str) -> list[str]:
    """Split text into lines, ignoring a trailing newline. Empty text yields an empty list."""
    if text == "":
        return []
    if text.endswith("\n"):
        text = text[:-1]
    return text.split("\n")


def render_create_record(
    *,
    path: str,
    new_revision: str,
    new_text: str,
    max_lines: int,
) -> str:
    lines = _split_into_lines(new_text)
    parts = [
        f"CREATED path={path}",
        f"revision: {new_revision}",
        f"line_count: 0 → {len(lines)}",
    ]
    if lines:
        parts.append("")
        parts.append(_render_diff_block("Added", 1, len(lines), lines, max_lines))
    return "\n".join(parts)


def render_edit_record(
    *,
    action: str,
    path: str,
    old_revision: str,
    new_revision: str,
    old_text: str,
    new_text: str,
    payload: dict,
    max_lines: int,
) -> str:
    """Render a frozen edit record describing an applied write.

    `Removed` line numbers reference the file before the edit; `Added` line numbers
    reference the file after.
    """
    old_lines = _split_into_lines(old_text)
    new_lines = _split_into_lines(new_text)
    parts = [
        f"EDITED path={path} action={action}",
        f"revision: {old_revision} → {new_revision}",
        f"line_count: {len(old_lines)} → {len(new_lines)}",
    ]
    inserted_text = payload.get("new_text") or ""
    inserted_lines = _split_into_lines(inserted_text)

    if action in ("replace_lines", "delete_lines"):
        rs = payload["start_line"]
        re = payload["end_line"]
        removed = old_lines[rs - 1 : re]
        parts.append("")
        parts.append(_render_diff_block("Removed", rs, re, removed, max_lines))

    if action in ("replace_lines", "insert_lines") and inserted_lines:
        added_start = payload["start_line"]
        added_end = added_start + len(inserted_lines) - 1
        parts.append("")
        parts.append(_render_diff_block("Added", added_start, added_end, inserted_lines, max_lines))

    # Show a small window of unchanged adjacent lines from the post-edit file so the model
    # can see boundary artifacts (duplicate fences, stray braces) that a Removed/Added pair
    # in isolation cannot reveal.
    context_window = 3
    context_before_end = payload["start_line"] - 1
    context_after_start = payload["start_line"] + len(inserted_lines)
    if context_before_end >= 1:
        context_before_start = max(1, context_before_end - context_window + 1)
        before_lines = new_lines[context_before_start - 1 : context_before_end]
        if before_lines:
            parts.append("")
            parts.append(
                _render_diff_block(
                    "Context before",
                    context_before_start,
                    context_before_end,
                    before_lines,
                    max_lines,
                )
            )
    if context_after_start <= len(new_lines):
        context_after_end = min(len(new_lines), context_after_start + context_window - 1)
        after_lines = new_lines[context_after_start - 1 : context_after_end]
        if after_lines:
            parts.append("")
            parts.append(
                _render_diff_block(
                    "Context after",
                    context_after_start,
                    context_after_end,
                    after_lines,
                    max_lines,
                )
            )

    return "\n".join(parts)


def render_live_window(
    *,
    path: str,
    revision: str,
    start_line: int,
    end_line: int,
    line_count: int,
    view_lines: list[str],
) -> str:
    """Render the canonical live-window output for a (path, revision, view) triple."""
    if line_count == 0:
        return f"FILE path={path} revision={revision} status=live line_count=0"
    header = f"FILE path={path} revision={revision} status=live lines={start_line}-{end_line} line_count={line_count}"
    if not view_lines:
        return header
    body = "\n".join(f"{i} | {line}" for i, line in enumerate(view_lines, start=start_line))
    return f"{header}\n{body}"


def render_tombstone(path: str, reason: str) -> str:
    return f"FILE path={path} status=stale [no longer accessible: {reason}]"


def render_view(
    text: str,
    start_line: int,
    max_lines: int,
    requested_end_line: int | None = None,
) -> tuple[int, int, list[str]]:
    """Return (end_line, line_count, view_lines) for a 1-based inclusive view from
    `start_line`, either up to `max_lines` lines or through `requested_end_line`,
    capped at the file's line count.

    `end_line` is the inclusive last line in the view, or `start_line - 1` if the view is
    empty (e.g. start_line is past EOF or the file is empty)."""
    lines = _split_into_lines(text)
    line_count = len(lines)
    if line_count == 0:
        return 0, 0, []
    start_idx = max(0, start_line - 1)
    if requested_end_line is None:
        end_idx = min(line_count, start_idx + max_lines)
    else:
        end_idx = min(line_count, requested_end_line)
    if start_idx >= end_idx:
        return start_line - 1, line_count, []
    return end_idx, line_count, lines[start_idx:end_idx]
