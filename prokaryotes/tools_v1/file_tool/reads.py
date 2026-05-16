import asyncio
import fcntl
import os
from pathlib import Path

from prokaryotes.tools_v1.file_tool.paths import (
    _open_text_file_no_follow,
    _raise_if_file_too_large,
)

# Per-path lock map shared across all FileTool instances in this process. Requests
# touching the same resolved path acquire the same asyncio.Lock and queue before
# entering asyncio.to_thread; the OS-level fcntl.flock inside threaded read/write
# transactions is the durable layer that survives multi-process worker setups. The
# map grows monotonically with unique touched paths; acceptable because the workspace
# path set is bounded and Lock objects are tiny.
_PATH_LOCKS: dict[str, asyncio.Lock] = {}


def _get_path_lock(path: str) -> asyncio.Lock:
    """Return the shared per-path asyncio.Lock for `path`, creating it on first use.

    get + setdefault is the right pattern here: dict.setdefault is atomic in CPython,
    so a lost create-race resolves to the same Lock instance for both callers.
    """
    lock = _PATH_LOCKS.get(path)
    if lock is None:
        lock = _PATH_LOCKS.setdefault(path, asyncio.Lock())
    return lock


def _locked_read_text(path: Path, max_file_bytes: int) -> str:
    """Synchronously read a text file under a shared advisory lock."""
    with _open_text_file_no_follow(path, os.O_RDONLY, "r") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_SH)
        _raise_if_file_too_large(fp.fileno(), path, max_file_bytes)
        return fp.read()


async def _read_text_under_file_tool_lock(path: Path, max_file_bytes: int) -> str:
    """Read `path` while participating in FileTool's same-path coordination.

    The asyncio lock prevents same-process readers from entering the thread pool while a
    FileTool writer for the same path is in its read-check-write critical section. The
    shared flock prevents cooperating readers in other processes from observing a file
    while a writer holds its exclusive lock.
    """
    path_lock = _get_path_lock(str(path))
    async with path_lock:
        return await asyncio.to_thread(_locked_read_text, path, max_file_bytes)
