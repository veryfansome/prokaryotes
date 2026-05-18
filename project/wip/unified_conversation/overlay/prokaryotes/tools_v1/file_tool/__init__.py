"""FileTool — operates on a `TurnItem` view via `view_provider`.

Replaces the `ContextPartition`-based constructor. The view is built by the
harness using `conversation_v1.project.current_turn_items(conversation,
historical_turns, active_turn)`, so the tool sees lifted live windows from
prior compactions in addition to the current turn's items.

The internal file I/O helpers (read_lines / create_file / write actions /
range / revision checks) are mechanically equivalent to upstream — only the
item type and the source of items change. Live-window refresh and tombstoning
operate on the view + pending-result list, exactly as before.

Lift logic (`lift_active_live_windows`, `_tool_round_start_index`) has been
removed from `live_windows.py` here — that selection now happens in
`ConversationCompactor._compute_lift_plan` against `TurnExecution`s rather
than against a flat item list. See the design doc.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import pathlib
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import prokaryotes

# Allow unchanged sibling modules (paths, reads, rendering, validation) in the real package
# to fall through. Only `live_windows.py` and `reconciliation.py` are overridden here.
_HERE = pathlib.Path(__file__).resolve().parent
for _parent_path in prokaryotes.__path__:
    _candidate = pathlib.Path(_parent_path).resolve() / "tools_v1" / "file_tool"
    if _candidate != _HERE and _candidate.is_dir() and str(_candidate) not in __path__:
        __path__.append(str(_candidate))

from prokaryotes.api_v1.models import FunctionToolCallback, ToolParameters, ToolSpec  # noqa: E402
from prokaryotes.conversation_v1.models import TurnItem  # noqa: E402
from prokaryotes.tools_v1.file_tool import live_windows, reads, reconciliation  # noqa: E402
from prokaryotes.tools_v1.file_tool.live_windows import (  # noqa: E402
    _annotation_requested_end_line,
    _is_unstable_coverage,
)
from prokaryotes.tools_v1.file_tool.paths import (  # noqa: E402
    FileToolFileTooLargeError,
    _open_text_file_no_follow,
    _raise_if_file_too_large,
    _resolve_path,
)
from prokaryotes.tools_v1.file_tool.rendering import (  # noqa: E402
    CURRENT_VIEW_MARKER_PREFIX,
    _append_live_window_refresh_note,
    _apply_line_edit,
    _count_lines,
    render_create_record,
    render_edit_record,
    render_live_window,
    render_view,
)
from prokaryotes.tools_v1.file_tool.validation import (  # noqa: E402
    _range_is_valid,
    _read_end_line,
    _read_start_line,
    _validate_create_payload,
    _validate_write_payload,
)

logger = logging.getLogger(__name__)

__all__ = ["FileTool", "reconcile_tracked_files"]


ViewProvider = Callable[[], list[TurnItem]]


class FileTool(FunctionToolCallback):
    """Read / create / edit UTF-8 text files with tracked live-window context.

    The `view_provider` is invoked on every `_refreshable_items()` call and
    must return a flat list of `TurnItem`s the tool may refresh — typically
    `current_turn_items(conversation, historical_turns, active_turn)`. Items
    just produced by this `FileTool` instance (still in flight for the current
    round) live in `_pending_result_items` until the LLM client commits them
    via `on_committed_turn_item`.
    """

    current_view_marker_prefix = CURRENT_VIEW_MARKER_PREFIX
    max_concurrent_reconcile_paths = 8
    max_file_bytes = 1_000_000
    max_lines = 200

    def __init__(
        self,
        view_provider: ViewProvider,
        workspace_root: Path | None = None,
    ):
        self._view_provider = view_provider
        self._workspace_root = workspace_root or Path.cwd()
        # Serializes call() so the read → mutate → refresh sequence is atomic per request.
        self._lock = asyncio.Lock()
        # Bridges "callback returned" → "view contains it". Concurrent tool calls in one
        # provider round each append their result here before returning so later refreshes
        # see prior in-flight results. Drained against the view at every _refreshable_items()
        # call: any item now present in the view is discarded from pending.
        self._pending_result_items: list[TurnItem] = []

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
                    result = await self._do_read_lines(call_id, resolved, payload)
                elif action == "create_file":
                    result = await self._do_create_file(call_id, resolved, payload)
                elif action in ("replace_lines", "insert_lines", "delete_lines"):
                    result = await self._do_write(call_id, resolved, action, payload)
                else:
                    result = _error_item(call_id, f"Unsupported action: {action!r}")
                self._pending_result_items.append(result)
                return result
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
        covering_window = self._find_covering_window(str(path), start_line, effective_requested_end_for_check)
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
        annotations = {
            "file_tool.path": str(path),
            "file_tool.revision": revision,
            "file_tool.status": "live",
            "file_tool.view_start_line": str(start_line),
            "file_tool.view_end_line": str(end_line),
        }
        if effective_requested_end_line is not None:
            annotations["file_tool.requested_end_line"] = str(effective_requested_end_line)
        return TurnItem(
            call_id=call_id,
            output=output,
            type="function_call_output",
            prokaryotes_annotations=annotations,
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
                live_windows._tombstone_live_windows(self._refreshable_items(), str(path), type(exc).__name__)
                return _error_item(call_id, f"{type(exc).__name__}: {exc}")

        refreshed_count = live_windows._refresh_live_windows(
            self._refreshable_items(), str(path), current_text, current_revision, self.max_lines
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
                live_windows._tombstone_live_windows(self._refreshable_items(), str(path), type(exc).__name__)
                return _error_item(call_id, f"{type(exc).__name__}: {exc}")
        live_windows._refresh_live_windows(
            self._refreshable_items(), str(path), current_text, current_revision, self.max_lines
        )
        return result_item

    def _find_covering_window(self, path: str, requested_start_line: int, requested_end_line: int) -> TurnItem | None:
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

    def _refreshable_items(self) -> list[TurnItem]:
        """Return the unified view + any in-flight tool results not yet in the view.

        Reads from `view_provider()` each call so the FileTool always sees the
        freshest snapshot of `current_turn_items` (lifted + historical + active turn).
        Pending results that have since been committed to the view are dropped.
        """
        view = self._view_provider()
        view_ids = {id(item) for item in view}
        self._pending_result_items = [item for item in self._pending_result_items if id(item) not in view_ids]
        return [*view, *self._pending_result_items]

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
            prokaryotes_annotations={"file_tool.path": str(path)},
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
            prokaryotes_annotations={"file_tool.path": str(path)},
        )
        return item, new_text, current_revision

    def _build_redundant_read_item(
        self,
        *,
        call_id: str,
        covering_window: TurnItem,
        path: Path,
        requested_end_line: int,
        requested_start_line: int,
    ) -> TurnItem:
        ann = covering_window.prokaryotes_annotations or {}
        view_start_line = int(ann["file_tool.view_start_line"])
        view_end_line = int(ann["file_tool.view_end_line"])
        intended_coverage_end = _annotation_requested_end_line(ann) or (view_start_line + self.max_lines - 1)
        revision = ann.get("file_tool.revision", "")
        output = "\n".join(
            [
                f"REDUNDANT_READ path={path} requested_lines={requested_start_line}-{requested_end_line}",
                (
                    "An existing live window already covers this span"
                    f" (rendered lines {view_start_line}-{view_end_line},"
                    f" intended coverage {view_start_line}-{intended_coverage_end},"
                    f" revision {revision}). Use that window."
                    f" To extend coverage, page forward from start_line={intended_coverage_end + 1}."
                ),
            ]
        )
        return TurnItem(
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
        annotations = {
            "file_tool.path": str(path),
            "file_tool.revision": current_revision,
            "file_tool.status": "live",
            "file_tool.view_start_line": str(view_start_line),
            "file_tool.view_end_line": str(end_line),
        }
        if requested_end_line is not None:
            annotations["file_tool.requested_end_line"] = str(requested_end_line)
        return TurnItem(
            call_id=call_id,
            output=output,
            type="function_call_output",
            prokaryotes_annotations=annotations,
        )

    @property
    def name(self) -> str:
        return "file_tool"

    @property
    def system_message_parts(self) -> list[str]:
        # Identical guidance to upstream; truncated here for brevity in the overlay.
        # The full text lives in the production module and doesn't change with this migration.
        return [
            f"## Using the `{self.name}` tool",
            "",
            f"- You SHOULD use the `{self.name}` tool to read, create, and edit UTF-8 text files.",
            (
                "- `read_lines` outputs are **live windows** the harness keeps synced to current disk"
                " state across turns."
            ),
            (
                "- `delete_lines` / `insert_lines` / `replace_lines` require `expected_revision` from"
                " a preceding `read_lines`."
            ),
            (
                "- `create_file` creates missing parent dirs inside the workspace; if the target already"
                " exists it returns a current live window so you can recover in one step."
            ),
            (
                f"- `read_lines` is capped at {self.max_lines} lines per call; spans over the cap return"
                " RANGE_TRUNCATED with a usable partial window and paging guidance."
            ),
            (
                "- Emit edits to the same file sequentially; concurrent edits to one path produce"
                " conflicts on all but one call."
            ),
        ]

    @property
    def tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Read line windows from a UTF-8 text file, create a new UTF-8 text file,"
                " or edit an existing UTF-8 text file by line range."
                " `read_lines` returns a numbered live window and a revision hash."
                " Write actions require `expected_revision`."
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
                        "description": "Path to the target file, resolved against the workspace root.",
                    },
                    "expected_revision": {
                        "type": ["string", "null"],
                        "description": "Required for write actions; null for read_lines / create_file.",
                    },
                    "start_line": {"type": ["integer", "null"], "minimum": 1, "description": "1-based start line."},
                    "end_line": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": "1-based inclusive end line.",
                    },
                    "new_text": {
                        "type": ["string", "null"],
                        "description": "UTF-8 text for create_file / replace_lines / insert_lines.",
                    },
                },
                required=["action", "end_line", "expected_revision", "new_text", "path", "start_line"],
            ),
        )


def _error_item(call_id: str, message: str) -> TurnItem:
    return TurnItem(call_id=call_id, output=f"ERROR {message}", type="function_call_output")


async def reconcile_tracked_files(
    items: list[TurnItem],
    workspace_root: Path | None = None,
) -> None:
    """Refresh live-window items in `items` against current on-disk content.

    Signature change: takes a flat `list[TurnItem]` (typically the unified view from
    `current_turn_items`) instead of a `ContextPartition`. Mutates items in place;
    tombstones live items for paths that are no longer accessible. Idempotent.
    """
    workspace_root = workspace_root or Path.cwd()
    paths_with_live: set[str] = set()
    for item in items:
        ann = item.prokaryotes_annotations or {}
        if item.type == "function_call_output" and ann.get("file_tool.status") == "live":
            path = ann.get("file_tool.path")
            if path:
                paths_with_live.add(path)
    if not paths_with_live:
        return
    semaphore = asyncio.Semaphore(FileTool.max_concurrent_reconcile_paths)

    async def reconcile_with_limit(path_str: str) -> None:
        async with semaphore:
            await reconciliation._reconcile_one_tracked_path(
                items,
                path_str,
                workspace_root,
                max_file_bytes=FileTool.max_file_bytes,
                max_lines=FileTool.max_lines,
            )

    await asyncio.gather(*(reconcile_with_limit(p) for p in paths_with_live))
