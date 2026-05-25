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
from uuid import uuid4

from prokaryotes.api_v1.models import FunctionToolCallback, ToolParameters, ToolSpec
from prokaryotes.conversation_v1.models import (
    TurnItem,
    WorkingFileSourceKind,
    WorkingFileWindow,
    coverage_eligible,
)
from prokaryotes.tools_v1.file_tool import live_windows, reads
from prokaryotes.tools_v1.file_tool.intervals import Interval, consolidate_intervals
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
        # Turn-local exposure tracking: window_id -> the revision whose content the model has actually been shown
        # this turn. This is distinct from `WorkingFileWindow.revision` (durable cached content): a per-read
        # `refresh_windows_for_path` advances a sibling window's `revision` mid-turn WITHOUT re-rendering it (the
        # provider loop never rebuilds the leading <working_files> block mid-turn), so durable-fresh ≠ shown. The
        # harness constructs FileTool *after* turn-start reconcile and *before* projection, so every current
        # window is about to be projected — seed it as exposed at its reconciled revision. Only a model-facing
        # render updates this map afterward (read FILE/RANGE_TRUNCATED output, empty-view marker, view-carrying
        # diagnostic); refresh must not. REDUNDANT_READ then requires exposed == current revision.
        self._exposed_window_revisions: dict[str, str | None] = {
            w.window_id: w.revision for w in working_file_provider()
        }

    def _mark_exposed(self, window_id: str, revision: str) -> None:
        """Record that the model has now been shown `window_id`'s content at `revision`."""
        self._exposed_window_revisions[window_id] = revision

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

        cap_end_line = start_line + self.max_lines - 1
        effective_req_end_for_check = requested_end_line if requested_end_line is not None else cap_end_line
        new_request = Interval(start_line, effective_req_end_for_check)

        # Read disk FIRST so the coverage check below runs against current content, not a stale cache.
        try:
            text = await reads._read_text_under_file_tool_lock(path, self.max_file_bytes)
        except (
            FileNotFoundError,
            FileToolFileTooLargeError,
            IsADirectoryError,
            PermissionError,
            UnicodeDecodeError,
        ) as exc:
            live_windows.tombstone_windows_for_path(self._windows(), str(path), type(exc).__name__)
            return _error_item(call_id, f"{type(exc).__name__}: {exc}")
        revision = sha256(text.encode("utf-8")).hexdigest()
        line_count = _count_lines(text)

        # The file is readable, so any tombstones for this path are stale. Snapshot which windows are diagnostics
        # BEFORE the refresh normalizes them to read_lines — a pre-existing diagnostic must not grant
        # REDUNDANT_READ for this read (it should fold via consolidation, recording this call), and refresh
        # erases the source_kind signal. Then refresh every surviving live window to the freshly-read revision.
        self._retire_tombstones_for_path(str(path))
        pre_refresh_diagnostic_ids = {
            w.window_id
            for w in self._windows()
            if w.path == str(path) and w.source_kind in live_windows.DIAGNOSTIC_SOURCE_KINDS
        }
        live_windows.refresh_windows_for_path(self._windows(), str(path), text, revision, self.max_lines)

        # Coverage check against known-fresh windows. A window grants REDUNDANT_READ only if it is
        # coverage-eligible (read_lines), was not a diagnostic before this refresh, AND its content at the current
        # revision has actually been shown to the model this turn (`_exposed_window_revisions[id] == revision`).
        # The exposure gate is the key: a coverage hit returns only a pointer, and the <working_files> block is a
        # turn-start artifact the provider loop never rebuilds mid-turn — so a window the refresh advanced to the
        # current revision but never re-rendered (e.g. a sibling refreshed by a disjoint read, or in-place
        # external change) must NOT short-circuit, or the model would act on stale content until next turn.
        # Such a read falls through and renders current content. A redundant read still records provenance.
        covering = next(
            (
                w
                for w in self._windows()
                if w.path == str(path)
                and coverage_eligible(w)
                and w.window_id not in pre_refresh_diagnostic_ids
                and self._exposed_window_revisions.get(w.window_id) == w.revision
                and _window_covers_request(w, new_request, self.max_lines)
            ),
            None,
        )
        if covering is not None:
            covering.origin_call_ids = sorted({*covering.origin_call_ids, call_id})
            return self._build_redundant_read_item(
                call_id=call_id, covering=covering, path=path, request=new_request, max_lines=self.max_lines
            )

        # Empty file or past-EOF read — skip consolidation (no content / no valid Interval). Mint an empty-view
        # placeholder preserving the requested start_line.
        if line_count == 0 or start_line > line_count:
            return self._mint_empty_view_window(
                call_id,
                path,
                revision,
                view_start_line=(1 if line_count == 0 else start_line),
                line_count=line_count,
            )

        # Clamp the request to what we'll render, then consolidate the reached windows around it.
        effective_req_end = min(effective_req_end_for_check, line_count)
        actual_view = Interval(start_line, min(cap_end_line, effective_req_end))
        consolidation_input_windows = [
            w
            for w in self._windows()
            if w.path == str(path) and w.status == "live" and w.view_end_line >= w.view_start_line
        ]
        existing_intervals = [Interval(w.view_start_line, w.view_end_line) for w in consolidation_input_windows]
        result = consolidate_intervals(existing_intervals, actual_view, self.max_lines)

        self._retire_windows(str(path), result.retired)
        self._mint_consolidated_windows(
            path=str(path),
            result=result,
            reached_windows=consolidation_input_windows,
            call_id=call_id,
            revision=revision,
            line_count=line_count,
            text=text,
            view=actual_view,
        )

        output = self._render_read_response(
            path=path,
            text=text,
            revision=revision,
            line_count=line_count,
            view=actual_view,
            requested_end_line=requested_end_line,
            cap_end_line=cap_end_line,
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
                live_windows.tombstone_windows_for_path(self._windows(), str(path), type(exc).__name__)
                return _error_item(call_id, f"{type(exc).__name__}: {exc}")

        # The write transaction read/wrote the file successfully, so the path is accessible — retire any stale
        # tombstones before refreshing. refresh skips stale windows, so a tombstone from a prior missing-file
        # read would otherwise survive on a now-recovered path.
        self._retire_tombstones_for_path(str(path))
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
                live_windows.tombstone_windows_for_path(self._windows(), str(path), type(exc).__name__)
                return _error_item(call_id, f"{type(exc).__name__}: {exc}")
        # A successful create (or recovered ALREADY_EXISTS read) proves the path is accessible — retire any stale
        # tombstones before refreshing, since refresh skips stale windows and they would otherwise survive.
        self._retire_tombstones_for_path(str(path))
        live_windows.refresh_windows_for_path(
            self._windows(),
            str(path),
            current_text,
            current_revision,
            self.max_lines,
            exclude_window_ids={call_id},
        )
        return result_item

    def _build_window_for_interval(
        self,
        *,
        path: str,
        interval: Interval,
        window_id: str,
        revision: str,
        line_count: int,
        text: str,
        origins: list[str],
    ) -> WorkingFileWindow:
        end_line, _line_count, view_lines = render_view(
            text, interval.start, self.max_lines, requested_end_line=interval.end
        )
        return WorkingFileWindow(
            window_id=window_id,
            path=path,
            status="live",
            revision=revision,
            rendered_output=render_live_window(
                path=path,
                revision=revision,
                start_line=interval.start,
                end_line=end_line,
                line_count=line_count,
                view_lines=view_lines,
            ),
            view_start_line=interval.start,
            view_end_line=end_line,
            requested_end_line=interval.end,
            line_count=line_count,
            origin_call_ids=origins,
            source_kind="read_lines",
        )

    def _mint_consolidated_windows(
        self,
        *,
        path: str,
        result,
        reached_windows: list[WorkingFileWindow],
        call_id: str,
        revision: str,
        line_count: int,
        text: str,
        view: Interval,
    ) -> None:
        # consolidate_intervals is pure (intervals only); reconstruct provenance here. The reached windows are
        # exactly those whose (view_start_line, view_end_line) matches a retired interval — union their origins
        # with the new call. Unioning over all matches is correct under a transient same-turn duplicate range.
        retired_ranges = {(i.start, i.end) for i in result.retired}
        reached_origins = sorted(
            {
                origin
                for w in reached_windows
                if (w.view_start_line, w.view_end_line) in retired_ranges
                for origin in w.origin_call_ids
            }
        )
        origins = sorted({call_id, *reached_origins})
        windows = self._windows()
        windows.append(
            self._build_window_for_interval(
                path=path,
                interval=result.primary,
                window_id=call_id,
                revision=revision,
                line_count=line_count,
                text=text,
                origins=origins,
            )
        )
        # Mark exposure ONLY when the primary's range exactly matches what the read rendered (`view` ==
        # actual_view). If consolidation extended the primary past the rendered range (it absorbed a window
        # reaching further, or merged a contiguous neighbor below the request), those extra lines were NOT shown
        # to the model — marking the whole primary exposed would let a later re-read of the unshown portion
        # wrongly return REDUNDANT_READ for content the model never saw. Secondaries are never rendered at all, so
        # they are never marked. Both unshown cases simply stay unexposed until next turn's projection re-seeds
        # them (cost: a same-turn re-read of that range re-renders instead of REDUNDANT — safe, just not deduped).
        if result.primary == view:
            self._mark_exposed(call_id, revision)
        for secondary in result.secondaries:
            secondary_id = f"wfw-{uuid4().hex}"
            windows.append(
                self._build_window_for_interval(
                    path=path,
                    interval=secondary,
                    window_id=secondary_id,
                    revision=revision,
                    line_count=line_count,
                    text=text,
                    origins=origins,
                )
            )

    def _mint_empty_view_window(
        self,
        call_id: str,
        path: Path,
        revision: str,
        *,
        view_start_line: int,
        line_count: int,
    ) -> TurnItem:
        windows = self._windows()
        if line_count == 0:
            # Empty file: retire every prior window for the path (minted on non-empty content).
            windows[:] = [w for w in windows if w.path != str(path)]
            view_start_line = 1
            view_end_line = 0
        else:
            # Past-EOF on a non-empty file: do NOT retire other windows; this view sits past content.
            view_end_line = view_start_line - 1
        output = render_live_window(
            path=str(path),
            revision=revision,
            start_line=view_start_line,
            end_line=view_end_line,
            line_count=line_count,
            view_lines=[],
        )
        windows.append(
            WorkingFileWindow(
                window_id=call_id,
                path=str(path),
                status="live",
                revision=revision,
                rendered_output=output,
                view_start_line=view_start_line,
                view_end_line=view_end_line,
                requested_end_line=view_end_line,
                line_count=line_count,
                origin_call_ids=[call_id],
                source_kind="read_lines",
            )
        )
        self._mark_exposed(call_id, revision)
        return TurnItem(
            call_id=call_id,
            output=output,
            type="function_call_output",
            prokaryotes_annotations=_file_tool_annotations(path, _PERSISTENCE_WORKING_FILE),
        )

    def _render_read_response(
        self,
        *,
        path: Path,
        text: str,
        revision: str,
        line_count: int,
        view: Interval,
        requested_end_line: int | None,
        cap_end_line: int,
    ) -> str:
        end_line, _line_count, view_lines = render_view(text, view.start, self.max_lines, requested_end_line=view.end)
        file_output = render_live_window(
            path=str(path),
            revision=revision,
            start_line=view.start,
            end_line=end_line,
            line_count=line_count,
            view_lines=view_lines,
        )
        if requested_end_line is not None and requested_end_line > cap_end_line and line_count > cap_end_line:
            remaining = min(requested_end_line, line_count) - cap_end_line
            header = "\n".join(
                [
                    (
                        f"RANGE_TRUNCATED path={path} requested_lines={view.start}-{requested_end_line}"
                        f" returned_lines={view.start}-{cap_end_line} line_count={line_count}"
                    ),
                    (
                        f"Your requested span exceeded the {self.max_lines}-line per-call cap."
                        f" The window below covers lines {view.start}-{cap_end_line}."
                        f" Call `read_lines` with `start_line={cap_end_line + 1}` to page through the"
                        f" remaining {remaining} lines."
                    ),
                ]
            )
            return f"{header}\n{file_output}"
        return file_output

    def _retire_tombstones_for_path(self, path: str) -> None:
        windows = self._windows()
        windows[:] = [w for w in windows if not (w.path == path and w.source_kind == "tombstone")]

    def _retire_windows(self, path: str, retired: list[Interval]) -> None:
        if not retired:
            return
        retired_ranges = {(i.start, i.end) for i in retired}
        windows = self._windows()
        windows[:] = [
            w for w in windows if not (w.path == path and (w.view_start_line, w.view_end_line) in retired_ranges)
        ]

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
        request: Interval,
        max_lines: int,
    ) -> TurnItem:
        # Pure message-building — the caller already appended call_id to covering.origin_call_ids. The variant
        # depends on why the window covers the request. EOF guidance applies ONLY when the window itself reaches
        # EOF (view_end_line >= line_count); a window that merely stopped at the per-call cap still has unread
        # content past it (line_count > cap), so the cap branch must win there even though request.end > line_count.
        cap = request.start + max_lines - 1
        fresh_effective_end = min(request.end, covering.line_count, cap)
        if covering.view_end_line >= covering.line_count and request.end > covering.line_count:
            body = (
                f"File ends at line {covering.line_count}; no content exists past that line. The existing window"
                " already covers up to EOF. Do not page forward past line_count."
            )
        elif request.end > cap and covering.line_count > cap:
            body = (
                f"An existing live window covers this span up to the per-call cap (line {fresh_effective_end})."
                f" To extend coverage, page forward from start_line={fresh_effective_end + 1}."
            )
        else:
            body = (
                "An existing live window already covers this range (rendered lines"
                f" {covering.view_start_line}-{covering.view_end_line}, revision {covering.revision or ''})."
                " Use that window."
            )
        output = "\n".join([f"REDUNDANT_READ path={path} requested_lines={request.start}-{request.end}", body])
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
    ) -> TurnItem:
        end_line, _, view_lines = render_view(current_text, view_start_line, self.max_lines)
        if line_count == 0:
            body = f"{self.current_view_marker_prefix}: empty file (line_count=0)"
        else:
            body_header = f"{self.current_view_marker_prefix} (lines {view_start_line}-{end_line} of {line_count}):"
            numbered = "\n".join(f"{i} | {line}" for i, line in enumerate(view_lines, start=view_start_line))
            body = body_header + ("\n" + numbered if numbered else "")
        output = "\n".join(header_lines + [body])
        # Mint a WorkingFileWindow for the diagnostic — its rendered_output carries the embedded current view; the
        # next reconcile pass normalizes source_kind back to `read_lines` and re-renders against fresh on-disk text.
        # `requested_end_line` is concrete (the rendered end) so reconcile re-renders a fixed extent — a `None`
        # would let it auto-expand on file growth and re-introduce overlap.
        self._windows().append(
            WorkingFileWindow(
                window_id=call_id,
                path=str(path),
                status="live",
                revision=current_revision,
                rendered_output=output,
                view_start_line=view_start_line,
                view_end_line=end_line,
                requested_end_line=end_line,
                line_count=line_count,
                origin_call_ids=[call_id],
                source_kind=source_kind,
            )
        )
        # The diagnostic output embeds a current view of the file, so the model is shown this content now.
        self._mark_exposed(call_id, current_revision)
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


def _window_covers_request(window: WorkingFileWindow, request: Interval, max_lines: int) -> bool:
    """Does `window`'s content satisfy `request` such that a fresh disk read would return no additional lines?

    Equivalent to: would a fresh `read_lines(request.start, request.end)` return its last line at or below
    `window.view_end_line`? The fresh read's effective end is clamped by the caller-requested end, by the file's
    `line_count` (EOF), and by the per-call cap `request.start + max_lines - 1`. Coverage runs after the per-read
    refresh, so `window.line_count` / `view_end_line` are the just-read values.
    """
    if request.start < window.view_start_line:
        return False
    fresh_effective_end = min(request.end, window.line_count, request.start + max_lines - 1)
    return fresh_effective_end <= window.view_end_line
