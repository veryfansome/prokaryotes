# File Tool

## Overview

`FileTool` gives the web harness LLMs a structured way to read line windows from files, create new UTF-8 text files, and edit existing text files by line range.

The feature is designed around one core idea: a `read_lines` output should not become a stale, historical snapshot the moment the file changes. Instead, `read_lines` outputs are treated as **live windows** into the current file. When the file changes later, either because the model edits it or because something else edits it on disk, the harness refreshes those older live windows in place so the next turn sees the current file state rather than an outdated copy.

This is a significant behavior change from a generic shell-based read/edit workflow:

- the model edits by line range instead of rewriting whole files by default
- every line edit is protected by `expected_revision`
- older reads for the same path stay synchronized with the file
- compaction avoids fossilizing live file contents into conversation summaries

Today the feature is wired into the Anthropic and OpenAI web harnesses. It is not yet registered in the script or eval harnesses.

---

## Why This Exists

The ordinary "read file, then edit based on what you saw" workflow breaks down quickly in an agentic chat setting:

- a file can change after the model reads it
- older tool outputs remain in the transcript and can mislead later turns
- line numbers in past edits drift after subsequent changes
- compaction can summarize stale file contents and make them impossible to refresh later

`FileTool` addresses those problems with a tracked-context model:

- reads become live windows
- writes become frozen audit records
- the harness reconciles tracked windows against disk before each turn
- compaction preserves file activity without summarizing live file bodies

---

## Core Concepts

### Live Windows

A `read_lines` result is a **live window**. It contains:

- the resolved file path
- the current file revision hash
- the requested starting line
- a bounded, line-numbered view of the file

Live windows are mutable `TurnItem`s in the active turn's unified view. On a later turn, the harness may rewrite:

- `output`
- `file_tool.revision`
- `file_tool.view_end_line`

That mutation happens when:

- a later `file_tool` write changes the same path
- an external edit changes the file between turns
- a transient diagnostic result needs to normalize back into an ordinary live window

Example:

```text
FILE path=/workspace/app.py revision=abc123 status=live lines=10-14 line_count=42
10 | def greet(name):
11 |     message = f"Hello, {name}"
12 |     return message
13 |
14 | print(greet("world"))
```

### Edit Records

Successful mutations do **not** become live windows. They become frozen **edit records**.

Edit records are historical audit trails. They record what changed at the time of the edit, but they are never refreshed later. Their line numbers are tied to the file state at the moment of that edit.

Example:

```text
EDITED path=/workspace/app.py action=replace_lines
revision: abc123 → def456
line_count: 42 → 42

Removed (lines 11-12):
11 |     message = f"Hello, {name}"
12 |     return message

Added (lines 11-12):
11 |     return f"Hello, {name}"
12 |

Context before (lines 8-10):
8 | # Friendly greeter
9 |
10 | def greet(name):

Context after (lines 13-15):
13 |
14 | print(greet("world"))
15 |

Live windows refreshed for this path: 1. Use current live windows for follow-up line numbers; this edit record is historical.
```

The Removed and Added blocks describe the change itself; the Context before and Context after blocks show up to three unchanged neighboring lines in the post-edit file so boundary artifacts like duplicate closing fences or stray braces are visible inline rather than only on a later live-window read. Either context block is omitted when nothing adjacent exists on that side — for example, edits that start at line 1 have no Context before, and edits that end at EOF have no Context after.

This distinction is important:

- live windows answer "what does the file look like now?"
- edit records answer "what changed during that earlier write, and what does the file look like immediately around the edit?"

### Tracked Metadata

Tracked state lives in `TurnItem.prokaryotes_annotations`.

Common keys:

| Key | Meaning |
|---|---|
| `file_tool.path` | Resolved absolute path for the file |
| `file_tool.revision` | SHA-256 of the current file text for live windows |
| `file_tool.status` | `live` or `stale` |
| `file_tool.view_start_line` | 1-based first line of the live window |
| `file_tool.view_end_line` | 1-based inclusive last line of the live window |
| `file_tool.requested_end_line` | Optional inclusive end line for an exact-span `read_lines`; omitted for open-ended paging reads |

These annotations are internal harness metadata. They round-trip through `TurnExecution` serialization, but provider payload serialization (the LLM-facing `ProjectedItem` projection) excludes them.
During live-window refresh, `file_tool.revision` and `file_tool.view_end_line` may change in place, while `file_tool.view_start_line` and `file_tool.requested_end_line` remain stable identity fields for that window.

`REDUNDANT_READ` items carry only `file_tool.path` — no `file_tool.status`, no `file_tool.revision`, no view-range annotations — so live-window refresh and tombstoning skip them (both gate on `status=live`) while compaction's path-activity check still treats their path as active.

### Reconciliation

Before each web-harness turn, the server builds the unified view of `TurnItem`s (lifted + historical-turn items + active-turn items) via `current_turn_items(conversation, historical_turns, active_turn)` and calls:

```python
await reconcile_tracked_files(items, workspace_root=Path.cwd())
```

That pass:

- finds every live window in the items list
- re-validates the stored path against the workspace root
- re-reads the current file contents
- refreshes any live windows whose revision is outdated
- tombstones windows whose files are no longer accessible

This is what lets the next turn see current file state even when the previous turn did not itself perform the edit. The refresh mutates the in-memory items in place; the `TurnExecution` documents persisted in Elasticsearch are immutable post-commit and remain at their original revisions.

It is also the portability contract for tracked files: any code path that loads `TurnExecution`s from Elasticsearch and wants current file state must run `reconcile_tracked_files(...)` over the resulting items before consuming live windows. The web harness already does this; future harnesses, search rendering, and debug tooling need to do the same or accept that tracked file views may lag disk state.

### Tombstones

If a previously tracked file can no longer be read safely, its live windows are converted to a **stale tombstone**:

```text
FILE path=/workspace/app.py status=stale [no longer accessible: FileNotFoundError]
```

Reasons include:

- file deleted
- path now points at a directory
- permission failure
- file became too large
- path escapes the workspace
- final path component became a symlink

Tombstones are not refreshed back into live windows automatically unless the model reads the file again successfully.

---

## Tool Contract

### Actions

`FileTool` supports five actions:

| Action | Purpose | Required fields |
|---|---|---|
| `read_lines` | Read a bounded, line-numbered view of a file | `path` |
| `create_file` | Create a new UTF-8 text file | `path`, `new_text` |
| `replace_lines` | Replace an inclusive line range | `path`, `expected_revision`, `start_line`, `end_line`, `new_text` |
| `insert_lines` | Insert lines before `start_line` | `path`, `expected_revision`, `start_line`, `new_text` |
| `delete_lines` | Delete an inclusive line range | `path`, `expected_revision`, `start_line`, `end_line` |

General conventions:

- line numbers are 1-based
- `end_line` is inclusive
- `insert_lines` may use `line_count + 1` to append at EOF
- `read_lines` defaults `start_line` to `1` when `null`
- `read_lines` may omit `end_line` to return up to `FileTool.max_lines` lines starting at `start_line`, or may supply `end_line` to request an exact inclusive range; ranges wider than `FileTool.max_lines` succeed partially and return a `RANGE_TRUNCATED` diagnostic plus a live window covering the first `FileTool.max_lines` lines of the requested span
- `create_file` requires `expected_revision`, `start_line`, and `end_line` to be `null`

### Read Lines Semantics

`read_lines` supports two modes:

- open-ended paging: provide `start_line` (or `null` for the start of the file) and omit `end_line`; the tool returns up to `FileTool.max_lines` lines, currently `200`
- exact span: provide both `start_line` and `end_line`; the tool returns that inclusive range, capped by EOF

When an exact-span request is wider than `FileTool.max_lines` and the file actually has content past the cap, the call succeeds partially and returns a `RANGE_TRUNCATED` diagnostic plus a live window covering the first `FileTool.max_lines` lines of the requested span, with explicit paging guidance for the remainder. When the requested span is wider than the cap but EOF falls within the cap (so nothing was clipped), the call returns an ordinary `FILE` view instead.

Before performing disk I/O, `read_lines` checks whether the requested span is already fully covered by an existing live window for the same path. Coverage is evaluated against the window's **intended coverage end** — `file_tool.requested_end_line` when annotated, otherwise `view_start_line + FileTool.max_lines - 1` — so an open-ended read whose `view_end_line` was clipped at EOF still recognizes follow-up reads of the same intended span as covered. When a covering window exists, the tool returns a short `REDUNDANT_READ` diagnostic that names the rendered range, intended coverage, the covering window's revision, and the `start_line` to page from to escape coverage. `RANGE_TRUNCATED` windows count as stable coverage for their returned span; `ALREADY_EXISTS`, `CONFLICT`, and `RANGE_ERROR` do not, since their diagnostic headers carry decision-relevant state the model still needs to react to.

Behavior details:

- empty file: returns `FILE ... status=live line_count=0`
- start line past EOF: returns a header with no numbered body
- exact-span `read_lines` calls preserve that same requested span on later live-window refreshes; for over-cap requests the effective span is pinned at `start_line + FileTool.max_lines - 1` so refreshes cannot grow the window past the cap on file growth
- file too large: returns an error item
- `read_lines` path is resolved relative to the workspace root if not absolute

### Create Semantics

`create_file` uses exclusive creation semantics.

- if the path does not exist, missing parent directories inside the workspace are created and a `CREATED` record is returned
- if the path already exists, the tool returns an `ALREADY_EXISTS` diagnostic plus a live window of the current file so the model can recover immediately

Example diagnostic shape:

```text
ALREADY_EXISTS path=/workspace/app.py current_revision=abc123
The file already exists. Read or edit the existing file instead.
Current view (lines 1-3 of 3):
1 | one
2 | two
3 | three
```

That diagnostic is itself a live window and will normalize into the canonical `FILE path=... status=live ...` form on the next refresh.

### Write Semantics

All line-edit actions require `expected_revision`.

The write transaction:

1. acquires the per-path in-process lock
2. opens the file without following a final symlink
3. takes an OS-level exclusive `flock`
4. reads the current file and computes its revision
5. rejects the edit if the revision no longer matches
6. validates the requested line range
7. applies the edit and writes the new content
8. refreshes older live windows for that path in the active turn's view
9. adds the actual refreshed live-window count to successful `EDITED` outputs

Two special non-success results also carry live windows:

- `CONFLICT` if `expected_revision` is stale
- `RANGE_ERROR` if the requested range is invalid for the current file

That lets the model retry immediately against current file state instead of needing an extra `read_lines` round trip. The read path uses a similar pattern for `RANGE_TRUNCATED` (described above): an over-cap exact-span request is answered with a usable partial window instead of a content-less error, so the model can page from `view_end_line + 1` in a single follow-up call.

The refresh count only includes live-window items that were actually rewritten. Windows already at the new revision are skipped and do not contribute to the count.

### Output Shapes

Common function-call output shapes:

| Shape | Meaning |
|---|---|
| `FILE path=... revision=... status=live ...` | Canonical live window |
| `CREATED path=...` | Successful create record |
| `EDITED path=... action=...` | Successful edit record with Removed/Added blocks and up to three lines of unchanged post-edit context above and below the edit, followed by the count of live windows actually refreshed |
| `ALREADY_EXISTS ...` + current view | Create collided with existing file |
| `CONFLICT ...` + current view | Revision mismatch during write |
| `RANGE_ERROR ...` + current view | Invalid line range for current file |
| `RANGE_TRUNCATED ...` + current view | `read_lines` exact span exceeded the per-call cap and content was clipped; window covers the first `FileTool.max_lines` lines with paging guidance for the remainder |
| `REDUNDANT_READ ...` | `read_lines` requested span was already fully covered by an existing live window; diagnostic points at that window instead of re-rendering the file |
| `FILE path=... status=stale [...]` | Tombstoned live window |
| `ERROR ...` | Validation or filesystem failure with no tracked recovery state |

---

## Request Lifecycle in the Web Harness

### 1. Conversation Sync

The harness first resolves the active `Conversation` snapshot for the request and loads the `TurnExecution`s for every bot message in the raw window (the same set the projection will interleave). It exposes the unified view of `TurnItem`s — lifted + historical + the in-flight active turn — through a `view_provider()` closure.

### 2. Tracked-File Reconciliation

Once the unified view is available, the harness runs `reconcile_tracked_files(view_provider(), workspace_root=...)`.

This is the point where:

- older live windows pick up external edits
- transient diagnostics normalize back into ordinary live windows
- inaccessible files are tombstoned

### 3. Tool Registration

The web harness then constructs:

```python
file_tool = FileTool(view_provider, workspace_root=Path.cwd())
```

The `view_provider` is called by `FileTool` on every action so same-turn writes can refresh earlier live windows in place before the provider serializes the next round. The unified view contains lifted items, historical-turn items, and the active turn's accumulating items.

### 4. Tool Guidance to the Model

The harness inserts FileTool-specific usage guidance into the system or developer message.

That guidance tells the model to:

- prefer `file_tool` over `shell_command` for normal reads and edits
- treat `read_lines` outputs as live windows
- treat write outputs as historical audit trails
- issue writes sequentially
- use `expected_revision` for all line edits
- page through larger files with later `start_line` values

### 5. Same-Turn Refresh

Provider SDKs may batch multiple tool calls in one round. `FileTool` keeps references to pending output items that have been returned but not yet appended to the active turn's items, so a later file-tool call in the same round can still refresh them.

This matters for patterns like:

- call `read_lines` for a file window
- edit same file
- call `read_lines` for a later window of the same file

All within one model turn.

Because live windows mutate earlier `function_call_output` items in place, provider-side prompt caches can lose reuse when a refreshed file window sits inside the cached prefix. In practice that means editing a tracked file may turn the next turn back into fresh input-token billing until a new stable prefix forms. Unchanged files do not churn cache state because windows are only re-rendered when the file revision changes or a transient diagnostic normalizes back into an ordinary `FILE ...` view.

---

## Safety Model

### Workspace Sandbox

Every path is resolved through `_resolve_path(path_arg, workspace_root)`.

Rules:

- relative paths are joined against `workspace_root`
- absolute paths are allowed only if they still resolve inside `workspace_root`
- any path that resolves outside the workspace is rejected

Reconciliation applies the same check again on later turns. This prevents a path that was valid earlier from escaping the workspace after a rename or symlink swap.

### No Final-Component Symlink Following

Low-level open helpers use `os.open(..., O_NOFOLLOW)` when supported.

This blocks the final path component from silently changing into a symlink target between validation and open. If that happens, the `read_lines` call or write fails and tracked live windows for that path are tombstoned rather than reading outside the workspace.

### Optimistic Concurrency

Line edits require the caller to supply `expected_revision`.

That protects against stale writes:

- if the file changed since the `read_lines` revision, the write does not apply
- instead the tool returns `CONFLICT` plus a fresh current view

This is simpler and safer than trying to merge line edits against unknown intervening changes.

`expected_revision` is only part of the story. The per-path lock plus OS-level `flock` are what make the read-check-write sequence atomic across cooperating same-host writers, so a second request reads after the first commit and returns `CONFLICT` instead of silently overwriting it.

### Per-Path Coordination

There are two coordination layers:

- process-local `asyncio.Lock` keyed by resolved path
- cross-process `fcntl.flock` inside the blocking read/write transaction

The in-process lock prevents same-worker races before operations enter the thread pool. The OS-level advisory lock protects cooperating readers and writers across processes on the same host.

Writers that bypass `file_tool` are outside that contract. Shell commands, editors, and unrelated processes that do not participate in these locks are observed only on the next non-covered `read_lines` call for that path (a covered re-read short-circuits to `REDUNDANT_READ` without touching disk) or on the next turn's reconciliation pass.

### Text and Newline Semantics

`FileTool` is UTF-8 text-only.

Behavior details:

- `create_file` writes `new_text` exactly as provided
- line-edit actions operate in Python text mode, so the first `file_tool` edit to a CRLF file rewrites it with LF line endings
- line edits preserve the existing trailing-newline policy for non-empty files
- inserting into a previously empty file writes a trailing newline even if `new_text` did not include one

### Content Limits

Current limits:

| Limit | Value |
|---|---|
| Max file size | 1,000,000 bytes |
| Max rendered lines per `read_lines` call | 200 |
| Max concurrent tracked-path reconciles per turn | 8 |

Files must be readable as UTF-8 text. Binary or oversized files are rejected.

### Crash Atomicity

Writes are protected against cooperating concurrent readers and writers, but they are not crash-atomic. The implementation rewrites files with `seek(0)`, `truncate()`, and `write()` under lock, so a process or host crash mid-write can leave a partial file that the next reconciliation will treat as current on-disk content.

---

## Compaction Integration

`FileTool` is tightly integrated with conversation compaction because live windows are mutable and normal conversation history is otherwise append-only.

### Why Compaction Needs Special Handling

If a live window body were copied into an ancestor summary, the file contents inside that summary could never be refreshed later. The summary would become a fossilized, stale copy of the file.

### Stripping Live-Window Bodies from Summary Input

Before summarization, compaction deep-copies the pre-tail bot turns' items and rewrites live-window outputs:

- canonical live windows are replaced with a path-only placeholder
- `ALREADY_EXISTS`, `CONFLICT`, `RANGE_ERROR`, and `RANGE_TRUNCATED` keep their diagnostic headers, but the embedded current-view body is replaced with the same placeholder

Edit records are left intact because they are historical and intentionally frozen.

Example placeholder:

```text
[Live tracked file: /workspace/app.py -- current contents are tracked via the live-window mechanism on subsequent turns, not summarized here.]
```

### Lifting Active Live Windows into the New Tail

When compaction creates a child `Conversation` snapshot, it lifts older live windows for any path that is still active in the recency tail. The lift step populates the child snapshot's `lifted_turn_items` field and anchors them at the first bot message in the raw tail that touched a relevant path (`lifted_anchor_source_id`).

Activity is keyed by `file_tool.path` annotations, including the frozen edit records produced by successful creates and edits and any `REDUNDANT_READ` diagnostics for the path. That lets a recent write or covered-read keep the relevant older live windows nearby even though the activity item itself is not refreshable.

Practical effect:

- if the model recently edited or referenced a file, the next active snapshot keeps the relevant live windows close to that activity
- the model does not have to rely only on old edit records with shifted line numbers
- lifted live windows carry their commit-time `file_tool.revision`; the next turn's `reconcile_tracked_files(...)` refreshes them in memory before projection

### Pending and Committed Child Snapshots

Compaction writes child snapshots through explicit `pending` and `committed` states with a `compaction_attempt_uuid`. That is primarily compaction infrastructure, but it matters for FileTool because `lifted_turn_items` (live windows + their paired `function_call`s) are part of the child snapshot state that must survive the CAS swap safely.

### Client Relabeling After Compaction

When a compaction swap commits, the compaction-status endpoint can return the child `snapshot_uuid`. The browser UI rewrites stored message-tree nodes from the old snapshot UUID to the child UUID so later follow-up turns and session resumes continue from the compacted child rather than an outdated parent pointer.

For deeper compaction details, see [Conversation Compaction](../compaction/README.md).

---

## UI Behavior

The browser UI formats `file_tool` activity entries into concise, user-readable summaries.

Examples:

- `read_lines` -> `Reading /path/to/file from line N`
- `create_file` -> `Creating /path/to/file`
- `replace_lines` -> `Editing /path/to/file lines A-B`
- `insert_lines` -> `Inserting at /path/to/file line N`
- `delete_lines` -> `Deleting /path/to/file lines A-B`

The formatter intentionally hides strict-mode `null` parameters so the activity tray does not show noisy `expected_revision: null` style output for `read_lines` and create calls.

Tool calls remain separate activity entries attached to assistant nodes; they are not rendered as ordinary conversation messages.

---

## Testing

The feature has coverage at several levels.

### Unit Tests

Key areas covered:

- action validation and output shapes
- live-window refresh after same-turn writes
- transient diagnostic normalization
- tombstoning when files disappear
- workspace escape protection
- symlink refusal
- provider-schema serialization quirks

Primary files:

- `tests/unit_tests/test_file_tool.py`
- `tests/unit_tests/test_turn_item_annotations.py`
- `tests/unit_tests/test_compaction_lift_plan.py`
- `tests/unit_tests/test_compaction_swap.py`
- `tests/unit_tests/test_strip_live_window_bodies.py`
- `tests/unit_tests/test_anthropic_v1.py`

### Integration Tests

Tier B coverage exercises the real web stack with fake LLM clients:

- multi-window reads
- refresh across writes and next-turn reconciliation
- create-file flow
- `ALREADY_EXISTS`, `CONFLICT`, and `RANGE_ERROR`
- missing-file tombstones
- symlink-escape tombstones
- live-window survival across compaction
- summary-input stripping for live windows

Primary file:

- `tests/integration_tests/tier_b/test_file_tool_flow.py`

### UI Tests

The client-side formatter for `file_tool` activity entries is covered in:

- `tests/ui_tests/file_tool_ui.test.js`

### Live Provider Smoke

Tier A passes with FileTool enabled in both the Anthropic and OpenAI web harnesses.

---

## Known Limitations

- Text only. The tool assumes UTF-8 and does not support binary patching.
- No rename/move/chmod operations. It is intentionally scoped to reads and line edits.
- Large files require paging or fallback tooling. Only 200 lines are rendered per view and files over 1 MB are rejected.
- Script and eval harnesses do not currently expose FileTool.
- A live window lifted during compaction can lag one refresh cycle if the file changed between the compaction snapshot and the Redis swap. The next turn's reconciliation repairs it.
- Cleanup for stale `pending` compaction children is deferred to later recovery/sweeper work. That is not a FileTool correctness gap, but it is related infrastructure.

---

## Relevant Code Files

| File | Role |
|---|---|
| `prokaryotes/tools_v1/file_tool/__init__.py` | `FileTool` class and the public `reconcile_tracked_files(items, workspace_root=...)` entry point. Delegates locked file I/O to `reads.py`, refresh/tombstoning to `live_windows.py`, path sandboxing to `paths.py`, and per-path reconciliation to `reconciliation.py`. |
| `prokaryotes/tools_v1/file_tool/live_windows.py` | Helpers operating on `file_tool.*` annotations: live-window detection, refresh, tombstoning, identity-tuple equality, recency-tail extraction, and summary-input stripping. |
| `prokaryotes/tools_v1/file_tool/paths.py` | Workspace sandboxing — `_resolve_path`, `_open_text_file_no_follow`, `_raise_if_file_too_large`, `FileToolFileTooLargeError`. |
| `prokaryotes/tools_v1/file_tool/reads.py` | Locked-read primitives (`_locked_read_text`, `_read_text_under_file_tool_lock`) and the shared per-path `asyncio.Lock` registry. |
| `prokaryotes/tools_v1/file_tool/reconciliation.py` | `_reconcile_one_tracked_path` — per-path reconcile helper invoked by `reconcile_tracked_files`. |
| `prokaryotes/tools_v1/file_tool/rendering.py` | `render_create_record`, `render_edit_record`, `render_live_window`, `render_tombstone`, `render_view`, line-edit text mutation, and the `CURRENT_VIEW_MARKER_PREFIX` constant. |
| `prokaryotes/tools_v1/file_tool/validation.py` | Payload field validation for `read_lines`, `create_file`, and the write actions. |
| `prokaryotes/conversation_v1/models.py` | `TurnItem.prokaryotes_annotations`. |
| `prokaryotes/conversation_v1/project.py` | `current_turn_items` — the unified `TurnItem` view (lifted + historical + active) consumed by `FileTool.view_provider`. |
| `prokaryotes/harness_v1/web.py` | Web-harness `FileTool` registration, `view_provider` wiring, and per-turn reconciliation. |
| `prokaryotes/context_v1/compaction.py` | `_compute_lift_plan` — module-level helper (run by `ConversationCompactor._prepare_compaction`) that lifts active live windows into the child snapshot's `lifted_turn_items`. |
| `scripts/static/ui.js` | Client-side rendering of `file_tool` activity entries |
| `tests/unit_tests/test_file_tool.py` | Core unit coverage |
| `tests/integration_tests/tier_b/test_file_tool_flow.py` | End-to-end Tier B regression coverage |
| `tests/ui_tests/file_tool_ui.test.js` | UI formatting coverage |
