import asyncio
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
from prokaryotes.tools_v1.file_tool import live_windows, reads, reconciliation
from prokaryotes.tools_v1.file_tool.live_windows import (
    _annotation_requested_end_line,
    _is_unstable_coverage,
)
from prokaryotes.tools_v1.file_tool.paths import (
    FileToolFileTooLargeError,
    _open_text_file_no_follow,
    _raise_if_file_too_large,
    _resolve_path,
)
from prokaryotes.tools_v1.file_tool.rendering import (
    CURRENT_VIEW_MARKER_PREFIX,
    _append_live_window_refresh_note,
    _apply_line_edit,
    _count_lines,
    render_create_record,
    render_edit_record,
    render_live_window,
    render_view,
)
from prokaryotes.tools_v1.file_tool.validation import (
    _range_is_valid,
    _read_end_line,
    _read_start_line,
    _validate_create_payload,
    _validate_write_payload,
)

logger = logging.getLogger(__name__)

__all__ = ["FileTool", "reconcile_tracked_files"]


class FileTool(FunctionToolCallback):
    """Tool to let the model read, create, and edit files with tracked context."""

    current_view_marker_prefix = CURRENT_VIEW_MARKER_PREFIX
    max_concurrent_reconcile_paths = 8
    max_file_bytes = 1_000_000
    max_lines = 200

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
                if action == "read_lines":
                    result = await self._do_read_lines(call_id, resolved, payload)
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

    async def _do_read_lines(
        self,
        call_id: str,
        path: Path,
        payload: dict,
    ) -> ContextPartitionItem:
        try:
            start_line = _read_start_line(payload)
            requested_end_line = _read_end_line(payload, start_line)
        except ValueError as exc:
            return _error_item(call_id, f"ValueError: {exc}")
        # Resolve the requested span for redundancy detection. Open-ended pages are evaluated
        # against the harness's intended coverage of [start_line, start_line + max_lines - 1]
        # so a re-read of a short file (where view_end_line was clipped at EOF) still detects
        # full coverage.
        if requested_end_line is None:
            effective_requested_end_for_check = start_line + self.max_lines - 1
        else:
            effective_requested_end_for_check = requested_end_line
        covering_window = self._find_covering_window(
            str(path),
            start_line,
            effective_requested_end_for_check,
        )
        if covering_window is not None:
            return self._build_redundant_read_item(
                call_id=call_id,
                covering_window=covering_window,
                path=path,
                requested_end_line=effective_requested_end_for_check,
                requested_start_line=start_line,
            )
        try:
            text = await reads._read_text_under_file_tool_lock(path, self.max_file_bytes)
        except (
            FileNotFoundError,
            FileToolFileTooLargeError,
            IsADirectoryError,
            PermissionError,
            UnicodeDecodeError,
        ) as exc:
            live_windows._tombstone_live_windows(self._refreshable_items(), str(path), type(exc).__name__)
            return _error_item(call_id, f"{type(exc).__name__}: {exc}")
        revision = sha256(text.encode("utf-8")).hexdigest()
        live_windows._refresh_live_windows(self._refreshable_items(), str(path), text, revision, self.max_lines)
        # Cap the effective end so live windows can never grow past max_lines on refresh,
        # even if the user-supplied span was wider. The original `requested_end_line` is
        # preserved separately so the RANGE_TRUNCATED diagnostic can echo what was asked.
        cap_end_line = start_line + self.max_lines - 1
        if requested_end_line is not None and requested_end_line > cap_end_line:
            effective_requested_end_line = cap_end_line
        else:
            effective_requested_end_line = requested_end_line
        line_count = _count_lines(text)
        if requested_end_line is not None and requested_end_line > cap_end_line and line_count > cap_end_line:
            # Cap actually clips live content. Return a RANGE_TRUNCATED view-carrying item
            # so the model gets a usable partial window plus an explicit paging instruction
            # for the remainder, instead of a content-less hard ERROR. The remainder is the
            # part of the *requested span* the model hasn't seen yet, not the rest of the
            # file — for a 1000-line file asked as 1-250, the remainder is 50 lines, not 800.
            remaining = min(requested_end_line, line_count) - cap_end_line
            return self._build_view_carrying_item(
                call_id=call_id,
                path=path,
                current_text=text,
                current_revision=revision,
                line_count=line_count,
                view_start_line=start_line,
                header_lines=[
                    (
                        f"RANGE_TRUNCATED path={path} requested_lines={start_line}-{requested_end_line}"
                        f" returned_lines={start_line}-{cap_end_line} line_count={line_count}"
                    ),
                    (
                        f"Your requested span exceeded the {self.max_lines}-line per-call cap."
                        f" The window below covers lines {start_line}-{cap_end_line}."
                        f" Call `read_lines` with `start_line={cap_end_line + 1}` to page through the"
                        f" remaining {remaining} lines."
                    ),
                ],
                requested_end_line=cap_end_line,
            )
        end_line, line_count, view_lines = render_view(
            text,
            start_line,
            self.max_lines,
            requested_end_line=effective_requested_end_line,
        )
        output = render_live_window(
            path=str(path),
            revision=revision,
            start_line=start_line,
            end_line=end_line,
            line_count=line_count,
            view_lines=view_lines,
        )
        annotations = {
            "file_tool.path": str(path),
            "file_tool.revision": revision,
            "file_tool.status": "live",
            "file_tool.view_start_line": str(start_line),
            "file_tool.view_end_line": str(end_line),
        }
        if effective_requested_end_line is not None:
            annotations["file_tool.requested_end_line"] = str(effective_requested_end_line)
        return ContextPartitionItem(
            call_id=call_id,
            output=output,
            type="function_call_output",
            prokaryotes_annotations=annotations,
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
                "expected_revision is required for write actions; call read_lines first to obtain it.",
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
                live_windows._tombstone_live_windows(self._refreshable_items(), str(path), type(exc).__name__)
                return _error_item(call_id, f"{type(exc).__name__}: {exc}")

        # Refresh runs after the path lock releases: it touches only this request's
        # in-memory partition, never disk, so it doesn't need the cross-request lock.
        # We always refresh — including on conflict and range-error results — so prior
        # live windows for this path move to the same revision the just-built result item
        # reports. On conflict that revision is the on-disk state the model just discovered;
        # on range-error and successful edits it is the post-transaction state. The
        # refresh is a no-op for windows already at this revision, so always-refresh is safe.
        refreshed_live_window_count = live_windows._refresh_live_windows(
            self._refreshable_items(),
            str(path),
            current_text,
            current_revision,
            self.max_lines,
        )
        if result_item.output and result_item.output.startswith("EDITED "):
            result_item.output = _append_live_window_refresh_note(
                result_item.output,
                refreshed_live_window_count,
            )
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
                live_windows._tombstone_live_windows(self._refreshable_items(), str(path), type(exc).__name__)
                return _error_item(call_id, f"{type(exc).__name__}: {exc}")

        live_windows._refresh_live_windows(
            self._refreshable_items(),
            str(path),
            current_text,
            current_revision,
            self.max_lines,
        )
        return result_item

    def _find_covering_window(
        self,
        path: str,
        requested_start_line: int,
        requested_end_line: int,
    ) -> ContextPartitionItem | None:
        """Return an existing live window whose intended coverage fully covers
        [requested_start_line, requested_end_line], or None.

        Intended coverage end is `file_tool.requested_end_line` when annotated; otherwise
        `view_start_line + max_lines - 1`. `view_end_line` alone understates coverage for
        open-ended reads that were clipped at EOF, so it isn't used as the lookup key."""
        for item in self._refreshable_items():
            ann = item.prokaryotes_annotations
            if not ann:
                continue
            if ann.get("file_tool.path") != path:
                continue
            if ann.get("file_tool.status") != "live":
                continue
            if _is_unstable_coverage(item):
                continue
            try:
                view_start_line = int(ann["file_tool.view_start_line"])
            except (KeyError, ValueError):
                continue
            intended_coverage_end = _annotation_requested_end_line(ann)
            if intended_coverage_end is None:
                intended_coverage_end = view_start_line + self.max_lines - 1
            if view_start_line <= requested_start_line and intended_coverage_end >= requested_end_line:
                return item
        return None

    def _refreshable_items(self) -> list[ContextPartitionItem]:
        partition_item_ids = {id(item) for item in self._partition.items}
        self._pending_result_items = [item for item in self._pending_result_items if id(item) not in partition_item_ids]
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
                        (
                            "The file changed since the revision returned by read_lines. "
                            "Use the current view before retrying."
                        ),
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

    def _build_redundant_read_item(
        self,
        *,
        call_id: str,
        covering_window: ContextPartitionItem,
        path: Path,
        requested_end_line: int,
        requested_start_line: int,
    ) -> ContextPartitionItem:
        """Build a REDUNDANT_READ diagnostic that points the model at an existing covering window.

        Carries `file_tool.path` only — no `file_tool.status=live`, no `file_tool.revision`, no
        rendered file body — so that `_refresh_live_windows` and `_tombstone_live_windows` skip
        it (they require `status=live`), while `_lift_active_live_windows` still treats the path
        as active (it only checks for `file_tool.path`)."""
        ann = covering_window.prokaryotes_annotations or {}
        view_start_line = int(ann["file_tool.view_start_line"])
        view_end_line = int(ann["file_tool.view_end_line"])
        intended_coverage_end = _annotation_requested_end_line(ann)
        if intended_coverage_end is None:
            intended_coverage_end = view_start_line + self.max_lines - 1
        revision = ann.get("file_tool.revision", "")
        output = "\n".join(
            [
                (f"REDUNDANT_READ path={path} requested_lines={requested_start_line}-{requested_end_line}"),
                (
                    "An existing live window already covers this span"
                    f" (rendered lines {view_start_line}-{view_end_line},"
                    f" intended coverage {view_start_line}-{intended_coverage_end},"
                    f" revision {revision})."
                    " Use that window. To extend coverage, page forward from"
                    f" start_line={intended_coverage_end + 1}."
                ),
            ]
        )
        return ContextPartitionItem(
            call_id=call_id,
            output=output,
            type="function_call_output",
            prokaryotes_annotations={"file_tool.path": str(path)},
        )

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
        requested_end_line: int | None = None,
    ) -> ContextPartitionItem:
        """Build a function_call_output that carries arbitrary header lines plus a fresh live-window
        view of the current file. Used for ALREADY_EXISTS, CONFLICT, RANGE_ERROR, and
        RANGE_TRUNCATED results so the model can immediately retry or page against the current
        revision. When `requested_end_line` is set, the view honours it and the annotation pins
        future refreshes to the same bound."""
        end_line, _, view_lines = render_view(
            current_text,
            view_start_line,
            self.max_lines,
            requested_end_line=requested_end_line,
        )
        if line_count == 0:
            body = f"{self.current_view_marker_prefix}: empty file (line_count=0)"
        else:
            body_header = f"{self.current_view_marker_prefix} (lines {view_start_line}-{end_line} of {line_count}):"
            numbered = "\n".join(f"{i} | {line}" for i, line in enumerate(view_lines, start=view_start_line))
            body = body_header + ("\n" + numbered if numbered else "")
        output = "\n".join(header_lines + [body])
        annotations = {
            "file_tool.path": str(path),
            "file_tool.revision": current_revision,
            "file_tool.status": "live",
            "file_tool.view_start_line": str(view_start_line),
            "file_tool.view_end_line": str(end_line),
        }
        if requested_end_line is not None:
            annotations["file_tool.requested_end_line"] = str(requested_end_line)
        return ContextPartitionItem(
            call_id=call_id,
            output=output,
            type="function_call_output",
            prokaryotes_annotations=annotations,
        )

    @classmethod
    def _get_path_lock(cls, path: str) -> asyncio.Lock:
        """Back-compat delegator to the per-path lock registry in `reads.py`. The canonical
        location is module-level state in `reads`; this classmethod is preserved because tests
        and any external callers still reach in via the FileTool class surface."""
        return reads._get_path_lock(path)

    @property
    def name(self) -> str:
        return "file_tool"

    @property
    def system_message_parts(self) -> list[str]:
        return [
            f"## Using the `{self.name}` tool",
            "",
            f"- You SHOULD use the `{self.name}` tool to read, create, and edit UTF-8 text files.",
            "  - It returns line-numbered views.",
            "  - It enforces optimistic concurrency on line edits.",
            (
                "  - Most importantly, it keeps prior reads in sync with current file state across turns, which allows"
                " you to hold **live windows** open on spans of lines, and not have to read the same section of a file"
                " more than once."
            ),
            (
                "- You SHOULD NOT use `shell_command` for routine file reads, creation, or edits. `shell_command` can"
                f" be used, however, if a file is outside of `{self.name}`'s limits or is a non-text or"
                " format-specific file, and you need to check metadata, identify file type, list archive contents,"
                " extract a small text excerpt with a specialized utility, or hand off to a format-specific tool."
            ),
            "- For UTF-8 text files, you SHOULD use:",
            "  - `create_file` to create a new file.",
            "  - `read_lines` to read a section of a file.",
            "  - `delete_lines`, `insert_lines`, or `replace_lines` to modify a section of a file.",
            (
                f"- `read_lines` is capped at {self.max_lines} lines per call. Page forward from an existing window's"
                " `view_end_line + 1` when you need more lines. An exact-span request wider than the cap succeeds"
                " partially: the tool returns a `RANGE_TRUNCATED` diagnostic plus a live window covering the first"
                f" {self.max_lines} lines of the requested span, and tells you which `start_line` to page from to"
                " cover the remainder."
            ),
            "- You SHOULD plan `read_lines` calls so each file is covered by clean, contiguous **live windows**.",
            "  - Avoid fragmented or overlapping ranges.",
            (
                "  - Do not call `read_lines` again for a span already covered by an existing live window in your"
                " current context; the tool will return a short `REDUNDANT_READ` diagnostic instead of re-reading."
            ),
            "  - If compaction removed that earlier live window from context, you may read that span again.",
            "- You MUST treat each `read_lines` output as a **live window**, and not as a static snapshot.",
            "  - Do not reread a span just because you suspect it may be stale.",
            (
                "  - Your harness keeps prior `read_lines` outputs synchronized with the current on-disk content of"
                " that span."
            ),
            (
                "  - After writes or external edits, earlier live windows remain authoritative for what the file looks"
                " like now."
            ),
            (
                "- You MUST treat each `create_file`, `delete_lines`, `insert_lines`, and `replace_lines` output as"
                " a frozen historical record, not a current view."
            ),
            (
                "  - In `delete_lines`, `insert_lines`, and `replace_lines` outputs, `Added` and `Removed` line"
                " numbers refer to the file state at the time of that edit."
            ),
            "  - After later edits, those line numbers may no longer point to the same content.",
            (
                "  - For follow-up edits, use the most recent `read_lines` output for that path instead of reusing"
                " line numbers from an edit record."
            ),
            (
                "- You MUST supply `expected_revision` on every `replace_lines`, `insert_lines`, and `delete_lines`"
                " call. Obtain it from a preceding `read_lines` output so the harness can detect concurrent on-disk"
                " changes."
            ),
            (
                "- `create_file` does not use `expected_revision`. It creates missing parent directories inside the"
                " workspace automatically, and if the final path already exists it returns a current live window so"
                " you can recover in one step."
            ),
            (
                "- You MUST emit file edits sequentially when multiple edits target the same file. Issue one write at"
                " a time and wait for its result before issuing the next. Concurrent writes against the same path will"
                " produce a conflict on all but one call, which you will then need to recover from."
            ),
            (
                f"- Files over {self.max_file_bytes} bytes are too large for `{self.name}`. If you only need metadata"
                " or a small excerpt, you MAY use a targeted `shell_command` instead."
            ),
        ]

    @property
    def tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Read line windows from a UTF-8 text file, create a new UTF-8 text file, or edit an existing UTF-8"
                " text file by line range. `read_lines` returns a numbered live window and a revision hash."
                " `replace_lines`, `insert_lines`, and `delete_lines` require the `expected_revision` from a"
                " preceding `read_lines` output for the same path."
            ),
            parameters=ToolParameters(
                properties={
                    "action": {
                        "type": "string",
                        "enum": ["read_lines", "create_file", "replace_lines", "insert_lines", "delete_lines"],
                        "description": "The file operation to perform.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the target file. Resolved against the workspace root.",
                    },
                    "expected_revision": {
                        "type": ["string", "null"],
                        "description": (
                            "Required for `replace_lines`, `insert_lines`, and `delete_lines`; pass null for"
                            " `read_lines` and `create_file`. Supply the revision from the most recent relevant"
                            " `read_lines` output for the same path. Used for optimistic concurrency."
                        ),
                    },
                    "start_line": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": (
                            "1-based start line. For `read_lines`, pass null to start at line 1. Omit `end_line` for"
                            " an open-ended page of up to 200 lines starting at `start_line`, or supply both"
                            " `start_line` and `end_line` for an exact inclusive span. When continuing from an"
                            " existing live window, the next page usually starts at its `view_end_line + 1`. Pass null"
                            " for `create_file`. For `replace_lines`, `insert_lines`, and `delete_lines`, this is the"
                            " first affected line. For `insert_lines`, lines are inserted before this line; pass"
                            " `line_count + 1` to append at EOF."
                        ),
                    },
                    "end_line": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": (
                            "1-based inclusive end line. For `read_lines`, pass null for an open-ended page or supply"
                            " an integer for an exact inclusive span. Spans wider than 200 lines succeed partially:"
                            " the call returns a `RANGE_TRUNCATED` diagnostic plus a live window covering the first"
                            " 200 lines of the requested span, with paging guidance for the remainder. Pass null for"
                            " `create_file` and `insert_lines`. Required for `replace_lines` and `delete_lines`."
                        ),
                    },
                    "new_text": {
                        "type": ["string", "null"],
                        "description": (
                            "UTF-8 text content for `create_file`, or replacement / insertion text for `replace_lines`"
                            " and `insert_lines`. Pass null for `read_lines` and `delete_lines`."
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


def _error_item(call_id: str, message: str) -> ContextPartitionItem:
    return ContextPartitionItem(
        call_id=call_id,
        output=f"ERROR {message}",
        type="function_call_output",
    )


def _locked_read_text(path: Path) -> str:
    """Back-compat wrapper preserving the original 1-arg signature for tests. Reads
    `FileTool.max_file_bytes` at call time so monkeypatched values propagate.
    """
    return reads._locked_read_text(path, FileTool.max_file_bytes)


async def _read_text_under_file_tool_lock(path: Path) -> str:
    """Back-compat wrapper preserving the original 1-arg signature. Reads
    `FileTool.max_file_bytes` at call time so monkeypatched values propagate.
    """
    return await reads._read_text_under_file_tool_lock(path, FileTool.max_file_bytes)


def _refresh_live_windows(
    items: list[ContextPartitionItem],
    path: str,
    text: str,
    revision: str,
) -> int:
    """Back-compat wrapper preserving the original 4-arg signature for tests. Reads
    `FileTool.max_lines` at call time so monkeypatched values propagate.
    """
    return live_windows._refresh_live_windows(items, path, text, revision, FileTool.max_lines)


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
            await reconciliation._reconcile_one_tracked_path(
                context_partition.items,
                path_str,
                workspace_root,
                max_file_bytes=FileTool.max_file_bytes,
                max_lines=FileTool.max_lines,
            )

    await asyncio.gather(*(reconcile_with_limit(path_str) for path_str in paths_with_live_items))
