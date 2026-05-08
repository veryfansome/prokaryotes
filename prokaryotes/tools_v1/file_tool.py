import asyncio
import errno
import fcntl
import json
import logging
import os
from hashlib import sha256
from pathlib import Path

from prokaryotes.api_v1.models import (
    ContextPartition,
    ContextPartitionItem,
    FunctionToolCallback,
    ToolParameters,
    ToolSpec,
)

logger = logging.getLogger(__name__)


class FileToolFileTooLargeError(ValueError):
    """Raised when a file exceeds FileTool's in-memory text processing limit."""


class FileTool(FunctionToolCallback):
    """Tool to let the model read, create, and edit files with tracked context."""

    current_view_marker_prefix = "Current view"
    max_concurrent_reconcile_paths = 8
    max_file_bytes = 1_000_000
    max_lines = 200

    # Class-level per-path locks shared across all FileTool instances in this process.
    # Requests touching the same resolved path acquire the same asyncio.Lock and queue
    # before entering asyncio.to_thread; the OS-level fcntl.flock inside threaded
    # read/write transactions is the durable layer that survives multi-process worker
    # setups. The map grows monotonically with unique touched paths; acceptable because
    # the workspace path set is bounded and Lock objects are tiny.
    _path_locks: dict[str, asyncio.Lock] = {}

    def __init__(
            self,
            context_partition: ContextPartition,
            workspace_root: Path | None = None,
    ):
        # context_partition is held by reference so write-side refresh and the harness's
        # reconciliation pass mutate the same items the LLM client will serialize next round.
        self._partition = context_partition
        self._workspace_root = workspace_root or Path.cwd()
        # Both LLM clients dispatch tool callbacks within a round concurrently; the lock
        # serializes call() to keep the read→mutate→refresh sequence atomic per request.
        self._lock = asyncio.Lock()
        # Callback result items are appended to ContextPartition only after the provider
        # client awaits the whole batch. Keep references here so later file_tool calls in
        # the same request can refresh live windows that have returned but are not appended yet.
        self._pending_result_items: list[ContextPartitionItem] = []

    async def call(self, arguments: str, call_id: str) -> ContextPartitionItem:
        async with self._lock:
            try:
                payload = json.loads(arguments)
            except Exception as exc:
                return _error_item(call_id, f"Invalid arguments JSON: {exc}")
            try:
                action = payload["action"]
                resolved = _resolve_path(payload["path"], self._workspace_root)
                if action == "read":
                    result = await self._do_read(call_id, resolved, payload)
                    self._pending_result_items.append(result)
                    return result
                if action == "create_file":
                    result = await self._do_create_file(call_id, resolved, payload)
                    self._pending_result_items.append(result)
                    return result
                if action in ("replace_lines", "insert_lines", "delete_lines"):
                    result = await self._do_write(call_id, resolved, action, payload)
                    self._pending_result_items.append(result)
                    return result
                result = _error_item(call_id, f"Unsupported action: {action!r}")
                self._pending_result_items.append(result)
                return result
            except Exception as exc:
                logger.exception("FileTool[%s] failed", call_id)
                return _error_item(call_id, f"{type(exc).__name__}: {exc}")

    async def _do_read(
            self,
            call_id: str,
            path: Path,
            payload: dict,
    ) -> ContextPartitionItem:
        try:
            start_line = _read_start_line(payload)
        except ValueError as exc:
            return _error_item(call_id, f"ValueError: {exc}")
        try:
            text = await _read_text_under_file_tool_lock(path)
        except (
                FileNotFoundError,
                FileToolFileTooLargeError,
                IsADirectoryError,
                PermissionError,
                UnicodeDecodeError,
        ) as exc:
            _tombstone_live_windows(self._refreshable_items(), str(path), type(exc).__name__)
            return _error_item(call_id, f"{type(exc).__name__}: {exc}")
        revision = sha256(text.encode("utf-8")).hexdigest()
        _refresh_live_windows(self._refreshable_items(), str(path), text, revision)
        end_line, line_count, view_lines = render_view(text, start_line, self.max_lines)
        output = render_live_window(
            path=str(path),
            revision=revision,
            start_line=start_line,
            end_line=end_line,
            line_count=line_count,
            view_lines=view_lines,
        )
        return ContextPartitionItem(
            call_id=call_id,
            output=output,
            type="function_call_output",
            prokaryotes_annotations={
                "file_tool.path": str(path),
                "file_tool.revision": revision,
                "file_tool.status": "live",
                "file_tool.view_start_line": str(start_line),
                "file_tool.view_end_line": str(end_line),
            },
        )

    async def _do_write(
            self,
            call_id: str,
            path: Path,
            action: str,
            payload: dict,
    ) -> ContextPartitionItem:
        expected_revision = payload.get("expected_revision")
        if not expected_revision:
            return _error_item(
                call_id,
                "expected_revision is required for write actions; read the file first to obtain it.",
            )
        validation_error = _validate_write_payload(action, payload)
        if validation_error:
            return _error_item(call_id, validation_error)

        # Two-layer cross-request safety. The class-level path lock serializes same-process
        # writers before they ever enter the thread pool; the OS-level fcntl.flock inside
        # _locked_write_transaction makes the read-check-write sequence atomic against any
        # cooperating writer in any process on this host. Without the path lock the read,
        # revision check, and write would race even within one Python process and the
        # second writer could pass an `expected_revision` check on stale-but-still-current-
        # at-read-time content and silently overwrite the first.
        path_lock = self._get_path_lock(str(path))
        async with path_lock:
            try:
                result_item, current_text, current_revision = await asyncio.to_thread(
                    self._locked_write_transaction,
                    call_id,
                    path,
                    action,
                    payload,
                    expected_revision,
                )
            except (
                    FileNotFoundError,
                    FileToolFileTooLargeError,
                    IsADirectoryError,
                    PermissionError,
                    UnicodeDecodeError,
            ) as exc:
                _tombstone_live_windows(self._refreshable_items(), str(path), type(exc).__name__)
                return _error_item(call_id, f"{type(exc).__name__}: {exc}")

        # Refresh runs after the path lock releases: it touches only this request's
        # in-memory partition, never disk, so it doesn't need the cross-request lock.
        # We always refresh — including on conflict and range-error results — so prior
        # live windows for this path move to the same revision the just-built result item
        # reports. On conflict that revision is the on-disk state the model just discovered;
        # on range-error and successful edits it is the post-transaction state. The
        # refresh is a no-op for windows already at this revision, so always-refresh is safe.
        _refresh_live_windows(self._refreshable_items(), str(path), current_text, current_revision)
        return result_item

    async def _do_create_file(
            self,
            call_id: str,
            path: Path,
            payload: dict,
    ) -> ContextPartitionItem:
        validation_error = _validate_create_payload(payload)
        if validation_error:
            return _error_item(call_id, validation_error)

        new_text = payload["new_text"]
        new_size = len(new_text.encode("utf-8"))
        if new_size > self.max_file_bytes:
            return _error_item(
                call_id,
                (
                    "FileToolFileTooLargeError: create would make "
                    f"{path} {new_size} bytes; limit is {self.max_file_bytes} bytes."
                ),
            )

        path_lock = self._get_path_lock(str(path))
        async with path_lock:
            try:
                result_item, current_text, current_revision = await asyncio.to_thread(
                    self._locked_create_transaction,
                    call_id,
                    path,
                    new_text,
                )
            except (
                    FileNotFoundError,
                    FileToolFileTooLargeError,
                    IsADirectoryError,
                    PermissionError,
                    UnicodeDecodeError,
            ) as exc:
                _tombstone_live_windows(self._refreshable_items(), str(path), type(exc).__name__)
                return _error_item(call_id, f"{type(exc).__name__}: {exc}")

        _refresh_live_windows(self._refreshable_items(), str(path), current_text, current_revision)
        return result_item

    def _refreshable_items(self) -> list[ContextPartitionItem]:
        partition_item_ids = {id(item) for item in self._partition.items}
        self._pending_result_items = [
            item for item in self._pending_result_items
            if id(item) not in partition_item_ids
        ]
        return [*self._partition.items, *self._pending_result_items]

    def _locked_write_transaction(
            self,
            call_id: str,
            path: Path,
            action: str,
            payload: dict,
            expected_revision: str,
    ) -> tuple[ContextPartitionItem, str, str]:
        """Run the read → revision-check → write critical section under fcntl.flock(LOCK_EX).

        Returns `(item, current_text, current_revision)` where `current_text` /
        `current_revision` describe the file's post-transaction on-disk state — the new
        content and revision for a successful edit, or the just-read content and revision
        for a conflict or range-error result. Conflict and range-error items are built from
        the just-read content so the caller can return them directly without further
        filesystem I/O. The caller uses `current_text` / `current_revision` to refresh
        any prior live windows in the partition; that refresh is a no-op when those
        windows are already at this revision.

        Runs synchronously inside `asyncio.to_thread`. The advisory lock is released when
        the file handle exits its `with` block and closes the file descriptor.
        """
        with _open_text_file_no_follow(path, os.O_RDWR, "r+") as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
            _raise_if_file_too_large(fp.fileno(), path, self.max_file_bytes)
            original_text = fp.read()
            current_revision = sha256(original_text.encode("utf-8")).hexdigest()
            line_count = _count_lines(original_text)

            if expected_revision != current_revision:
                item = self._build_view_carrying_item(
                    call_id=call_id,
                    path=path,
                    current_text=original_text,
                    current_revision=current_revision,
                    line_count=line_count,
                    view_start_line=payload.get("start_line") or 1,
                    header_lines=[
                        f"CONFLICT path={path} expected_revision={expected_revision}"
                        f" current_revision={current_revision}",
                        "The file changed since the revision you read. Re-read before retrying.",
                    ],
                )
                return item, original_text, current_revision
            if not _range_is_valid(action, payload, line_count):
                item = self._build_view_carrying_item(
                    call_id=call_id,
                    path=path,
                    current_text=original_text,
                    current_revision=current_revision,
                    line_count=line_count,
                    view_start_line=payload.get("start_line") or 1,
                    header_lines=[
                        f"RANGE_ERROR path={path} action={action} current_revision={current_revision}",
                        f"Requested line range is out of bounds for line_count={line_count}.",
                    ],
                )
                return item, original_text, current_revision

            updated_text = _apply_line_edit(original_text, action, payload)
            updated_size = len(updated_text.encode("utf-8"))
            if updated_size > self.max_file_bytes:
                return (
                    _error_item(
                        call_id,
                        (
                            "FileToolFileTooLargeError: edit would make "
                            f"{path} {updated_size} bytes; limit is {self.max_file_bytes} bytes."
                        ),
                    ),
                    original_text,
                    current_revision,
                )
            fp.seek(0)
            fp.truncate()
            fp.write(updated_text)
            fp.flush()
            new_revision = sha256(updated_text.encode("utf-8")).hexdigest()

        edit_output = render_edit_record(
            action=action,
            path=str(path),
            old_revision=current_revision,
            new_revision=new_revision,
            old_text=original_text,
            new_text=updated_text,
            payload=payload,
            max_lines=self.max_lines,
        )
        # Edit records carry only `file_tool.path` — they are frozen audit trails, never
        # refreshed, but the path annotation lets compaction recognize this path as active.
        item = ContextPartitionItem(
            call_id=call_id,
            output=edit_output,
            type="function_call_output",
            prokaryotes_annotations={"file_tool.path": str(path)},
        )
        return item, updated_text, new_revision

    def _locked_create_transaction(
            self,
            call_id: str,
            path: Path,
            new_text: str,
    ) -> tuple[ContextPartitionItem, str, str]:
        try:
            # `create_file` is allowed to materialize a missing directory tree as long as
            # the resolved target path stays inside the workspace sandbox.
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "x", encoding="utf-8") as fp:
                fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
                fp.write(new_text)
                fp.flush()
        except FileExistsError:
            with _open_text_file_no_follow(path, os.O_RDONLY, "r") as fp:
                fcntl.flock(fp.fileno(), fcntl.LOCK_SH)
                _raise_if_file_too_large(fp.fileno(), path, self.max_file_bytes)
                current_text = fp.read()
            current_revision = sha256(current_text.encode("utf-8")).hexdigest()
            line_count = _count_lines(current_text)
            item = self._build_view_carrying_item(
                call_id=call_id,
                path=path,
                current_text=current_text,
                current_revision=current_revision,
                line_count=line_count,
                view_start_line=1,
                header_lines=[
                    f"ALREADY_EXISTS path={path} current_revision={current_revision}",
                    "The file already exists. Read or edit the existing file instead.",
                ],
            )
            return item, current_text, current_revision

        current_revision = sha256(new_text.encode("utf-8")).hexdigest()
        item = ContextPartitionItem(
            call_id=call_id,
            output=render_create_record(
                path=str(path),
                new_revision=current_revision,
                new_text=new_text,
                max_lines=self.max_lines,
            ),
            type="function_call_output",
            prokaryotes_annotations={"file_tool.path": str(path)},
        )
        return item, new_text, current_revision

    def _build_view_carrying_item(
            self,
            *,
            call_id: str,
            path: Path,
            current_text: str,
            current_revision: str,
            line_count: int,
            view_start_line: int,
            header_lines: list[str],
    ) -> ContextPartitionItem:
        """Build a function_call_output that carries arbitrary header lines plus a fresh live-window
        view of the current file. Used for ALREADY_EXISTS, CONFLICT, and RANGE_ERROR results so
        the model can immediately retry against the current revision."""
        end_line, _, view_lines = render_view(current_text, view_start_line, self.max_lines)
        if line_count == 0:
            body = f"{self.current_view_marker_prefix}: empty file (line_count=0)"
        else:
            body_header = (
                f"{self.current_view_marker_prefix} "
                f"(lines {view_start_line}-{end_line} of {line_count}):"
            )
            numbered = "\n".join(
                f"{i} | {line}" for i, line in enumerate(view_lines, start=view_start_line)
            )
            body = body_header + ("\n" + numbered if numbered else "")
        output = "\n".join(header_lines + [body])
        return ContextPartitionItem(
            call_id=call_id,
            output=output,
            type="function_call_output",
            prokaryotes_annotations={
                "file_tool.path": str(path),
                "file_tool.revision": current_revision,
                "file_tool.status": "live",
                "file_tool.view_start_line": str(view_start_line),
                "file_tool.view_end_line": str(end_line),
            },
        )

    @classmethod
    def _get_path_lock(cls, path: str) -> asyncio.Lock:
        """Return the shared per-path asyncio.Lock for `path`, creating it on first use.

        get + setdefault is the right pattern here: dict.setdefault is atomic in CPython,
        so a lost create-race resolves to the same Lock instance for both callers.
        """
        lock = cls._path_locks.get(path)
        if lock is None:
            lock = cls._path_locks.setdefault(path, asyncio.Lock())
        return lock

    @property
    def name(self) -> str:
        return "file_tool"

    @property
    def system_message_parts(self) -> list[str]:
        return [
            f"## Using the `{self.name}` tool",
            (
                f"- Use `{self.name}` instead of `shell_command` for routine file reads, file"
                " creation, and edits. It returns line-numbered views, enforces optimistic"
                " concurrency on line edits, and keeps prior reads in sync with the current file"
                " state across turns."
            ),
            (
                "- Prefer `create_file` for new files and targeted line-range operations"
                " (`replace_lines`, `insert_lines`, `delete_lines`) over whole-file rewrites."
            ),
            (
                f"- Treat each `{self.name}` read output as a **live window** into the file, not a"
                " static snapshot. The harness keeps every prior read window for a file in sync"
                " with the current on-disk content: when a subsequent write or external edit"
                " changes the file, earlier read outputs in your conversation history are updated"
                " in-place so their rendered views and revisions reflect the current file content."
                " Earlier windows are therefore authoritative for what the file looks like *now*,"
                " not what it looked like at the time of the read."
            ),
            (
                f"- Treat each `{self.name}` write output (an **edit record**) as a frozen"
                " historical audit trail, not a current view. The line numbers in an edit record's"
                " `Removed` and `Added` blocks refer to the file state at the time of *that* edit."
                " After subsequent edits to the same file, those absolute line numbers may have"
                " shifted and no longer point to the same content. To target the same content for"
                " further edits, always consult the most recent live window for the path; never"
                " carry line numbers forward from an edit record."
            ),
            (
                "- Emit file edits sequentially: issue one write at a time and wait for its result"
                " before issuing the next, especially when multiple edits target the same file."
                " Concurrent writes against the same path will produce a conflict on all but one,"
                " which you will then have to recover from."
            ),
            (
                "- `create_file` does not use `expected_revision`; it creates missing parent"
                " directories inside the workspace automatically, and if the final path already"
                " exists it returns the current file view so you can recover in one step."
            ),
            (
                "- All line-edit actions require `expected_revision`. Obtain it from a preceding"
                " `read` and pass it on every write so the harness can detect concurrent on-disk"
                " changes."
            ),
            (
                f"- Read views are capped at {self.max_lines} lines per call; for larger files,"
                " issue additional reads at later `start_line` values."
            ),
            (
                f"- Files over {self.max_file_bytes} bytes are too large for `{self.name}`;"
                " use targeted shell commands if you only need metadata or small excerpts."
            ),
        ]

    @property
    def tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Read a file, create a new UTF-8 text file, or edit an existing file by line"
                " range. Reads return a numbered view and a revision hash; line edits"
                " (replace_lines / insert_lines / delete_lines) require the expected_revision"
                " from a preceding read for optimistic concurrency."
            ),
            parameters=ToolParameters(
                properties={
                    "action": {
                        "type": "string",
                        "enum": ["read", "create_file", "replace_lines", "insert_lines", "delete_lines"],
                        "description": "The file operation to perform.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the target file. Resolved against the workspace root.",
                    },
                    "expected_revision": {
                        "type": ["string", "null"],
                        "description": (
                            "Required for line-edit actions; pass null for `read` and `create_file`."
                            " The revision returned by a preceding read call. Used for optimistic"
                            " concurrency."
                        ),
                    },
                    "start_line": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": (
                            "1-based start line. For `read`, pass null to read from the start of"
                            " the file. Pass null for `create_file`. For line edits, this is the"
                            " first affected line. For `insert_lines`, lines are inserted *before*"
                            " this line; pass `line_count + 1` to append at EOF."
                        ),
                    },
                    "end_line": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": (
                            "1-based inclusive end line. Pass null for `read`, `create_file`, and"
                            " `insert_lines`. Required for `replace_lines` and `delete_lines`."
                        ),
                    },
                    "new_text": {
                        "type": ["string", "null"],
                        "description": (
                            "File contents for `create_file`, or replacement / insertion text for"
                            " `replace_lines` and `insert_lines`. Pass null for `read` and"
                            " `delete_lines`."
                        ),
                    },
                },
                required=[
                    "action",
                    "end_line",
                    "expected_revision",
                    "new_text",
                    "path",
                    "start_line",
                ],
            ),
        )


async def reconcile_tracked_files(
        context_partition: ContextPartition,
        workspace_root: Path | None = None,
) -> None:
    """Refresh live windows in `context_partition.items` against current on-disk content.

    Called by each harness's `post_chat()` after `sync_context_partition()`. Tombstones
    every live item for a path that is no longer accessible. Idempotent: items already at
    the current revision are left untouched.
    """
    workspace_root = workspace_root or Path.cwd()
    paths_with_live_items: set[str] = set()
    for item in context_partition.items:
        ann = item.prokaryotes_annotations or {}
        if item.type == "function_call_output" and ann.get("file_tool.status") == "live":
            path = ann.get("file_tool.path")
            if path:
                paths_with_live_items.add(path)

    semaphore = asyncio.Semaphore(FileTool.max_concurrent_reconcile_paths)

    async def reconcile_with_limit(path_str: str) -> None:
        async with semaphore:
            await _reconcile_one_tracked_path(
                context_partition.items,
                path_str,
                workspace_root,
            )

    await asyncio.gather(*(reconcile_with_limit(path_str) for path_str in paths_with_live_items))


async def _reconcile_one_tracked_path(
        items: list[ContextPartitionItem],
        path_str: str,
        workspace_root: Path,
) -> None:
    try:
        path = _resolve_path(path_str, workspace_root)
        current_text = await _read_text_under_file_tool_lock(path)
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
    _refresh_live_windows(items, path_str, current_text, current_revision)


async def _read_text_under_file_tool_lock(path: Path) -> str:
    """Read `path` while participating in FileTool's same-path coordination.

    The asyncio lock prevents same-process readers from entering the thread pool while a
    FileTool writer for the same path is in its read-check-write critical section. The
    shared flock prevents cooperating readers in other processes from observing a file
    while a writer holds its exclusive lock.
    """
    path_lock = FileTool._get_path_lock(str(path))
    async with path_lock:
        return await asyncio.to_thread(_locked_read_text, path)


def _locked_read_text(path: Path) -> str:
    """Synchronously read a text file under a shared advisory lock."""
    with _open_text_file_no_follow(path, os.O_RDONLY, "r") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_SH)
        _raise_if_file_too_large(fp.fileno(), path, FileTool.max_file_bytes)
        return fp.read()


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
        raise FileToolFileTooLargeError(
            f"{path} is {size} bytes; limit is {max_file_bytes} bytes."
        )


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
        removed = old_lines[rs - 1:re]
        parts.append("")
        parts.append(_render_diff_block("Removed", rs, re, removed, max_lines))

    if action in ("replace_lines", "insert_lines") and inserted_lines:
        added_start = payload["start_line"]
        added_end = added_start + len(inserted_lines) - 1
        parts.append("")
        parts.append(_render_diff_block("Added", added_start, added_end, inserted_lines, max_lines))

    return "\n".join(parts)


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
    header = (
        f"FILE path={path} revision={revision} status=live"
        f" lines={start_line}-{end_line} line_count={line_count}"
    )
    if not view_lines:
        return header
    body = "\n".join(f"{i} | {line}" for i, line in enumerate(view_lines, start=start_line))
    return f"{header}\n{body}"


def render_tombstone(path: str, reason: str) -> str:
    return f"FILE path={path} status=stale [no longer accessible: {reason}]"


def _read_start_line(payload: dict) -> int:
    start_line = payload.get("start_line")
    if start_line is None:
        return 1
    if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
        raise ValueError("start_line for read must be null or an integer >= 1")
    return start_line


def render_view(text: str, start_line: int, max_lines: int) -> tuple[int, int, list[str]]:
    """Return (end_line, line_count, view_lines) for a 1-based inclusive view from
    `start_line` up to `max_lines` lines, capped at the file's line count.

    `end_line` is the inclusive last line in the view, or `start_line - 1` if the view is
    empty (e.g. start_line is past EOF or the file is empty)."""
    lines = _split_into_lines(text)
    line_count = len(lines)
    if line_count == 0:
        return 0, 0, []
    start_idx = max(0, start_line - 1)
    end_idx = min(line_count, start_idx + max_lines)
    if start_idx >= end_idx:
        return start_line - 1, line_count, []
    return end_idx, line_count, lines[start_idx:end_idx]


def _apply_line_edit(text: str, action: str, payload: dict) -> str:
    lines = _split_into_lines(text)
    inserted = _split_into_lines(payload.get("new_text") or "")
    start = payload["start_line"]
    if action == "replace_lines":
        end = payload["end_line"]
        result = lines[:start - 1] + inserted + lines[end:]
    elif action == "insert_lines":
        result = lines[:start - 1] + inserted + lines[start - 1:]
    elif action == "delete_lines":
        end = payload["end_line"]
        result = lines[:start - 1] + lines[end:]
    else:
        raise ValueError(f"Unsupported write action: {action!r}")
    trailing_newline = text.endswith("\n") or (text == "" and len(result) > 0)
    return "\n".join(result) + ("\n" if trailing_newline and result else "")


def _count_lines(text: str) -> int:
    return len(_split_into_lines(text))


def _error_item(call_id: str, message: str) -> ContextPartitionItem:
    return ContextPartitionItem(
        call_id=call_id,
        output=f"ERROR {message}",
        type="function_call_output",
    )


def _validate_write_payload(action: str, payload: dict) -> str | None:
    start_line = payload.get("start_line")
    if not _is_positive_int(start_line):
        return f"start_line is required for {action} and must be an integer >= 1."
    if action in ("replace_lines", "delete_lines"):
        end_line = payload.get("end_line")
        if not _is_positive_int(end_line):
            return f"end_line is required for {action} and must be an integer >= 1."
        if start_line > end_line:
            return f"start_line must be <= end_line for {action}."
    if action in ("replace_lines", "insert_lines"):
        new_text = payload.get("new_text")
        if not isinstance(new_text, str) or new_text == "":
            return f"new_text is required for {action} and must be a non-empty string."
    return None


def _validate_create_payload(payload: dict) -> str | None:
    if payload.get("expected_revision") is not None:
        return "expected_revision must be null for create_file."
    if payload.get("start_line") is not None:
        return "start_line must be null for create_file."
    if payload.get("end_line") is not None:
        return "end_line must be null for create_file."
    new_text = payload.get("new_text")
    if not isinstance(new_text, str):
        return "new_text is required for create_file and must be a string."
    return None


def _is_positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 1


def _range_is_valid(action: str, payload: dict, line_count: int) -> bool:
    start = payload.get("start_line")
    end = payload.get("end_line")
    if action == "insert_lines":
        if not _is_positive_int(start):
            return False
        # start_line in [1, line_count + 1]; end_line is unused.
        return 1 <= start <= line_count + 1
    if action in ("replace_lines", "delete_lines"):
        if not _is_positive_int(start) or not _is_positive_int(end):
            return False
        if line_count == 0:
            return False
        return 1 <= start <= end <= line_count
    return False


def _refresh_live_windows(
        items: list[ContextPartitionItem],
        path: str,
        text: str,
        revision: str,
) -> None:
    """Re-render every live window for `path` against `text` / `revision`. Items already
    at `revision` are left alone. Shared by `FileTool` writes and `reconcile_tracked_files`."""
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
        end_line, line_count, view_lines = render_view(text, start_line, FileTool.max_lines)
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


def _has_transient_file_diagnostic(item: ContextPartitionItem) -> bool:
    output = item.output or ""
    return (
        output.startswith("ALREADY_EXISTS ")
        or output.startswith("CONFLICT ")
        or output.startswith("RANGE_ERROR ")
    )


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


def _resolve_path(path_arg: str, workspace_root: Path) -> Path:
    """Resolve `path_arg` against `workspace_root` and verify it does not escape it.

    Absolute paths are kept as-is; relative paths are joined against `workspace_root`. The
    resolved path must lie within `workspace_root.resolve()`."""
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
        raise ValueError(
            f"Path {path_arg!r} escapes workspace root {workspace_root}"
        ) from exc
    return resolved


def _split_into_lines(text: str) -> list[str]:
    """Split text into lines, ignoring a trailing newline. Empty text yields an empty list."""
    if text == "":
        return []
    if text.endswith("\n"):
        text = text[:-1]
    return text.split("\n")
