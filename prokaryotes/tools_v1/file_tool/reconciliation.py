from hashlib import sha256
from pathlib import Path

from prokaryotes.api_v1.models import ContextPartitionItem
from prokaryotes.tools_v1.file_tool.live_windows import (
    _refresh_live_windows,
    _tombstone_live_windows,
)
from prokaryotes.tools_v1.file_tool.paths import (
    FileToolFileTooLargeError,
    _resolve_path,
)
from prokaryotes.tools_v1.file_tool.reads import _read_text_under_file_tool_lock


async def _reconcile_one_tracked_path(
    items: list[ContextPartitionItem],
    path_str: str,
    workspace_root: Path,
    *,
    max_file_bytes: int,
    max_lines: int,
) -> None:
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
        _tombstone_live_windows(items, path_str, type(exc).__name__)
        return

    current_revision = sha256(current_text.encode("utf-8")).hexdigest()
    _refresh_live_windows(items, path_str, current_text, current_revision, max_lines)
