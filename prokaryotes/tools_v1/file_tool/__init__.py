"""FileTool — first-class working-file state.

The tool mutates `Conversation.working_file_windows` directly via `working_file_provider`. Read-like outputs
(read_lines, RANGE_TRUNCATED, ALREADY_EXISTS, CONFLICT, RANGE_ERROR, REDUNDANT_READ) are annotated
`file_tool.persistence="working_file"` so projection drops them from later turns — their durable relevance lives
in `working_file_windows`. Frozen edit records (CREATED, EDITED) are annotated `file_tool.persistence="history"`
and ride the transcript forward as ordinary history.
"""

from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

import asyncio
import fcntl
import json
import logging
import os
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

from prokaryotes.api_v1.models import FunctionToolCallback, ToolParameters, ToolSpec
from prokaryotes.conversation_v1.models import (
    TurnItem,
    WorkingFileSourceKind,
    WorkingFileWindow,
    coverage_eligible,
)
from prokaryotes.tools_v1.file_tool import live_windows, reads
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

__all__ = ["FileTool"]


WorkingFileProvider = Callable[[], list[WorkingFileWindow]]

_PERSISTENCE_WORKING_FILE = "working_file"
_PERSISTENCE_HISTORY = "history"
_PERSISTENCE_ANNOTATION = "file_tool.persistence"
_PATH_ANNOTATION = "file_tool.path"


def _file_tool_annotations(path: Path, persistence: str) -> dict[str, str]:
    """Annotations applied to every file-tool function_call_output.

    `file_tool.persistence` drives projection's historical-output filter (working_file outputs are dropped on
    later turns; history outputs ride forward). `file_tool.path` is the absolute resolved path the call acted on;
    branch divergence and cold rebuild read it off kept TurnExecutions to compute active paths even when the call
    didn't mint a new `WorkingFileWindow` (successful edits refresh existing windows in place; REDUNDANT_READ
    points at an existing window without minting one).
    """
    return {_PERSISTENCE_ANNOTATION: persistence, _PATH_ANNOTATION: str(path)}


class FileTool(FunctionToolCallback):
    """Read / create / edit UTF-8 text files; persist live file context as `WorkingFileWindow`s.

    The `working_file_provider` callable returns the mutable backing list (typically
    `conversation.working_file_windows`). The tool refreshes, normalizes, and mints windows in place. Subsequent
    `FileTool` calls in the same turn see the updated state directly; the next turn's reconcile pass refreshes
    everything against disk before any new call.
    """

    current_view_marker_prefix = CURRENT_VIEW_MARKER_PREFIX
    max_concurrent_reconcile_paths = 8
    max_file_bytes = 1_000_000
    max_lines = 200

    def __init__(
        self,
        working_file_provider: WorkingFileProvider,
        workspace_root: Path | None = None,
    ):
        self._working_file_provider = working_file_provider
        self._workspace_root = workspace_root or Path.cwd()
        # Serializes call() so the read → mutate → refresh sequence is atomic per request.
        self._lock = asyncio.Lock()

    async def call(self, arguments: str, call_id: str) -> TurnItem:
        async with self._lock:
            try:
                payload = json.loads(arguments)
            except Exception as exc:
                return _error_item(call_id, f"Invalid arguments JSON: {exc}")
            try:
                action = payload["action"]
                resolved = _resolve_path(payload["path"], self._workspace_root)
                if action == "read_lines":
                    return await self._do_read_lines(call_id, resolved, payload)
                if action == "create_file":
                    return await self._do_create_file(call_id, resolved, payload)
                if action in ("replace_lines", "insert_lines", "delete_lines"):
                    return await self._do_write(call_id, resolved, action, payload)
                return _error_item(call_id, f"Unsupported action: {action!r}")
            except Exception as exc:
                logger.exception("FileTool[%s] failed", call_id)
                return _error_item(call_id, f"{type(exc).__name__}: {exc}")

    async def _do_read_lines(self, call_id: str, path: Path, payload: dict) -> TurnItem:
        try:
            start_line = _read_start_line(payload)
            requested_end_line = _read_end_line(payload, start_line)
        except ValueError as exc:
            return _error_item(call_id, f"ValueError: {exc}")
        effective_requested_end_for_check = (
            start_line + self.max_lines - 1 if requested_end_line is None else requested_end_line
        )
        covering = self._find_covering_window(str(path), start_line, effective_requested_end_for_check)
        if covering is not None:
            return self._build_redundant_read_item(
                call_id=call_id,
                covering=covering,
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
            live_windows.tombstone_windows_for_path(
                self._windows(), str(path), type(exc).__name__
            )
            return _error_item(call_id, f"{type(exc).__name__}: {exc}")
        revision = sha256(text.encode("utf-8")).hexdigest()
        live_windows.refresh_windows_for_path(self._windows(), str(path), text, revision, self.max_lines)
        cap_end_line = start_line + self.max_lines - 1
        effective_requested_end_line = (
            cap_end_line if requested_end_line is not None and requested_end_line > cap_end_line else requested_end_line
        )
        line_count = _count_lines(text)
        if requested_end_line is not None and requested_end_line > cap_end_line and line_count > cap_end_line:
            remaining = min(requested_end_line, line_count) - cap_end_line
            return self._build_view_carrying_item(
                call_id=call_id,
                path=path,
                current_text=text,
                current_revision=revision,
                line_count=line_count,
                view_start_line=start_line,
                source_kind="range_truncated",
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
            text, start_line, self.max_lines, requested_end_line=effective_requested_end_line
        )
        output = render_live_window(
            path=str(path),
            revision=revision,
            start_line=start_line,
            end_line=end_line,
            line_count=line_count,
            view_lines=view_lines,
        )
        self._windows().append(
            WorkingFileWindow(
                window_id=call_id,
                path=str(path),
                status="live",
                revision=revision,
                rendered_output=output,
                view_start_line=start_line,
                view_end_line=end_line,
                requested_end_line=effective_requested_end_line,
                source_kind="read_lines",
            )
        )
        return TurnItem(
            call_id=call_id,
            output=output,
            type="function_call_output",
            prokaryotes_annotations=_file_tool_annotations(path, _PERSISTENCE_WORKING_FILE),
        )

    async def _do_write(self, call_id: str, path: Path, action: str, payload: dict) -> TurnItem:
        expected_revision = payload.get("expected_revision")
        if not expected_revision:
            return _error_item(
                call_id,
                "expected_revision is required for write actions; call read_lines first to obtain it.",
            )
        validation_error = _validate_write_payload(action, payload)
        if validation_error:
            return _error_item(call_id, validation_error)

        path_lock = reads._get_path_lock(str(path))
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
                live_windows.tombstone_windows_for_path(
                    self._windows(), str(path), type(exc).__name__
                )
                return _error_item(call_id, f"{type(exc).__name__}: {exc}")

        refreshed_count = live_windows.refresh_windows_for_path(
            self._windows(),
            str(path),
            current_text,
            current_revision,
            self.max_lines,
            exclude_window_ids={call_id},
        )
        if result_item.output and result_item.output.startswith("EDITED "):
            result_item.output = _append_live_window_refresh_note(result_item.output, refreshed_count)
        return result_item

    async def _do_create_file(self, call_id: str, path: Path, payload: dict) -> TurnItem:
        validation_error = _validate_create_payload(payload)
        if validation_error:
            return _error_item(call_id, validation_error)
        new_text = payload["new_text"]
        new_size = len(new_text.encode("utf-8"))
        if new_size > self.max_file_bytes:
            return _error_item(
                call_id,
                (
                    f"FileToolFileTooLargeError: create would make {path} {new_size} bytes;"
                    f" limit is {self.max_file_bytes} bytes."
                ),
            )
        path_lock = reads._get_path_lock(str(path))
        async with path_lock:
            try:
                result_item, current_text, current_revision = await asyncio.to_thread(
                    self._locked_create_transaction, call_id, path, new_text
                )
            except (
                FileNotFoundError,
                FileToolFileTooLargeError,
                IsADirectoryError,
                PermissionError,
                UnicodeDecodeError,
            ) as exc:
                live_windows.tombstone_windows_for_path(
                    self._windows(), str(path), type(exc).__name__
                )
                return _error_item(call_id, f"{type(exc).__name__}: {exc}")
        live_windows.refresh_windows_for_path(
            self._windows(),
            str(path),
            current_text,
            current_revision,
            self.max_lines,
            exclude_window_ids={call_id},
        )
        return result_item

    def _find_covering_window(
        self, path: str, requested_start_line: int, requested_end_line: int
    ) -> WorkingFileWindow | None:
        for window in self._windows():
            if window.path != path:
                continue
            if not coverage_eligible(window):
                continue
            intended_coverage_end = window.requested_end_line or (window.view_start_line + self.max_lines - 1)
            if window.view_start_line <= requested_start_line and intended_coverage_end >= requested_end_line:
                return window
        return None

    def _windows(self) -> list[WorkingFileWindow]:
        return self._working_file_provider()

    def _locked_write_transaction(
        self, call_id: str, path: Path, action: str, payload: dict, expected_revision: str
    ) -> tuple[TurnItem, str, str]:
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
                    source_kind="conflict",
                    header_lines=[
                        (
                            f"CONFLICT path={path} expected_revision={expected_revision}"
                            f" current_revision={current_revision}"
                        ),
                        (
                            "The file changed since the revision returned by read_lines."
                            " Use the current view before retrying."
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
                    source_kind="range_error",
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
                            f"FileToolFileTooLargeError: edit would make {path} {updated_size} bytes;"
                            f" limit is {self.max_file_bytes} bytes."
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
        item = TurnItem(
            call_id=call_id,
            output=edit_output,
            type="function_call_output",
            prokaryotes_annotations=_file_tool_annotations(path, _PERSISTENCE_HISTORY),
        )
        return item, updated_text, new_revision

    def _locked_create_transaction(self, call_id: str, path: Path, new_text: str) -> tuple[TurnItem, str, str]:
        try:
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
                source_kind="already_exists",
                header_lines=[
                    f"ALREADY_EXISTS path={path} current_revision={current_revision}",
                    "The file already exists. Read or edit the existing file instead.",
                ],
            )
            return item, current_text, current_revision
        current_revision = sha256(new_text.encode("utf-8")).hexdigest()
        item = TurnItem(
            call_id=call_id,
            output=render_create_record(
                path=str(path), new_revision=current_revision, new_text=new_text, max_lines=self.max_lines
            ),
            type="function_call_output",
            prokaryotes_annotations=_file_tool_annotations(path, _PERSISTENCE_HISTORY),
        )
        return item, new_text, current_revision

    def _build_redundant_read_item(
        self,
        *,
        call_id: str,
        covering: WorkingFileWindow,
        path: Path,
        requested_end_line: int,
        requested_start_line: int,
    ) -> TurnItem:
        intended_coverage_end = covering.requested_end_line or (covering.view_start_line + self.max_lines - 1)
        output = "\n".join(
            [
                f"REDUNDANT_READ path={path} requested_lines={requested_start_line}-{requested_end_line}",
                (
                    "An existing live window already covers this span"
                    f" (rendered lines {covering.view_start_line}-{covering.view_end_line},"
                    f" intended coverage {covering.view_start_line}-{intended_coverage_end},"
                    f" revision {covering.revision or ''}). Use that window."
                    f" To extend coverage, page forward from start_line={intended_coverage_end + 1}."
                ),
            ]
        )
        return TurnItem(
            call_id=call_id,
            output=output,
            type="function_call_output",
            prokaryotes_annotations=_file_tool_annotations(path, _PERSISTENCE_WORKING_FILE),
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
        source_kind: WorkingFileSourceKind,
        header_lines: list[str],
        requested_end_line: int | None = None,
    ) -> TurnItem:
        end_line, _, view_lines = render_view(
            current_text, view_start_line, self.max_lines, requested_end_line=requested_end_line
        )
        if line_count == 0:
            body = f"{self.current_view_marker_prefix}: empty file (line_count=0)"
        else:
            body_header = f"{self.current_view_marker_prefix} (lines {view_start_line}-{end_line} of {line_count}):"
            numbered = "\n".join(f"{i} | {line}" for i, line in enumerate(view_lines, start=view_start_line))
            body = body_header + ("\n" + numbered if numbered else "")
        output = "\n".join(header_lines + [body])
        # Mint a WorkingFileWindow for the diagnostic — its rendered_output carries the embedded current view; the
        # next reconcile pass normalizes source_kind back to `read_lines` and re-renders against fresh on-disk text.
        self._windows().append(
            WorkingFileWindow(
                window_id=call_id,
                path=str(path),
                status="live",
                revision=current_revision,
                rendered_output=output,
                view_start_line=view_start_line,
                view_end_line=end_line,
                requested_end_line=requested_end_line,
                source_kind=source_kind,
            )
        )
        return TurnItem(
            call_id=call_id,
            output=output,
            type="function_call_output",
            prokaryotes_annotations=_file_tool_annotations(path, _PERSISTENCE_WORKING_FILE),
        )

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


def _error_item(call_id: str, message: str) -> TurnItem:
    return TurnItem(call_id=call_id, output=f"ERROR {message}", type="function_call_output")
