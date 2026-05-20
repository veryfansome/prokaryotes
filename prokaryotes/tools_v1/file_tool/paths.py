import errno
import os
from pathlib import Path


class FileToolFileTooLargeError(ValueError):
    """Raised when a file exceeds FileTool's in-memory text processing limit."""


def _open_text_file_no_follow(path: Path, flags: int, mode: str):
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags | nofollow_flag)
    except OSError as exc:
        if nofollow_flag and exc.errno == errno.ELOOP:
            raise PermissionError(f"Refusing to follow symlink for {path}") from exc
        raise
    try:
        return os.fdopen(fd, mode, encoding="utf-8")
    except Exception:
        os.close(fd)
        raise


def _raise_if_file_too_large(fd: int, path: Path, max_file_bytes: int) -> None:
    size = os.fstat(fd).st_size
    if size > max_file_bytes:
        raise FileToolFileTooLargeError(f"{path} is {size} bytes; limit is {max_file_bytes} bytes.")


def _resolve_path(path_arg: str, workspace_root: Path) -> Path:
    """Resolve `path_arg` against `workspace_root` and verify it does not escape it.

    Absolute paths are kept as-is; relative paths are joined against `workspace_root`. The resolved path must lie
    within `workspace_root.resolve()`."""
    if not isinstance(path_arg, str) or not path_arg:
        raise ValueError("path is required and must be a non-empty string")
    candidate = Path(path_arg)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    resolved = candidate.resolve()
    workspace_resolved = workspace_root.resolve()
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError as exc:
        raise ValueError(f"Path {path_arg!r} escapes workspace root {workspace_root}") from exc
    return resolved
