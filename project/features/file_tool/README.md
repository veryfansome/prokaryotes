# File Tool

## Overview

`FileTool` gives the web harness LLMs a structured way to read files, create new UTF-8 text files, and edit existing text files by line range.

The feature is designed around one core idea: a file read should not become a stale, historical snapshot the moment the file changes. Instead, read outputs are treated as **live windows** into the current file. When the file changes later, either because the model edits it or because something else edits it on disk, the harness refreshes those older read outputs in place so the next turn sees the current file state rather than an outdated copy.

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

A `read` result is a **live window**. It contains:

- the resolved file path
- the current file revision hash
- the requested starting line
- a bounded, line-numbered view of the file

Live windows are mutable partition items. On a later turn, the harness may rewrite:

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
revision: abc123 -> def456
line_count: 42 -> 42

Removed (lines 11-12):
11 |     message = f"Hello, {name}"
12 |     return message

Added (lines 11-12):
11 |     return f"Hello, {name}"
12 |
```

This distinction is important:

- live windows answer "what does the file look like now?"
- edit records answer "what changed during that earlier write?"

### Tracked Metadata

Tracked state lives in `ContextPartitionItem.prokaryotes_annotations`.

Common keys:

| Key | Meaning |
|---|---|
| `file_tool.path` | Resolved absolute path for the file |
| `file_tool.revision` | SHA-256 of the current file text for live windows |
| `file_tool.status` | `live` or `stale` |
| `file_tool.view_start_line` | 1-based first line of the live window |
| `file_tool.view_end_line` | 1-based inclusive last line of the live window |

These annotations are internal harness metadata. They round-trip through partition serialization, but provider payload serialization excludes them.

### Reconciliation

Before each web-harness turn, the server calls:

```python
await reconcile_tracked_files(context_partition, workspace_root=Path.cwd())
```

That pass:

- finds every live window in the active partition
- re-validates the stored path against the workspace root
- re-reads the current file contents
- refreshes any live windows whose revision is outdated
- tombstones windows whose files are no longer accessible

This is what lets the next turn see current file state even when the previous turn did not itself perform the edit.

It is also the portability contract for tracked files: any code path that loads a `ContextPartition` from Redis or Elasticsearch and wants current file state must run `reconcile_tracked_files(...)` before consuming live windows. The web harnesses already do this; future harnesses, search rendering, and debug tooling need to do the same or accept that tracked file views may lag disk state.

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
| `read` | Read a bounded, line-numbered view of a file | `path` |
| `create_file` | Create a new UTF-8 text file | `path`, `new_text` |
| `replace_lines` | Replace an inclusive line range | `path`, `expected_revision`, `start_line`, `end_line`, `new_text` |
| `insert_lines` | Insert lines before `start_line` | `path`, `expected_revision`, `start_line`, `new_text` |
| `delete_lines` | Delete an inclusive line range | `path`, `expected_revision`, `start_line`, `end_line` |

General conventions:

- line numbers are 1-based
- `end_line` is inclusive
- `insert_lines` may use `line_count + 1` to append at EOF
- `read` defaults `start_line` to `1` when `null`
- `create_file` requires `expected_revision`, `start_line`, and `end_line` to be `null`

### Read Semantics

`read` returns at most `FileTool.max_lines` lines, currently `200`.

Behavior details:

- empty file: returns `FILE ... status=live line_count=0`
- start line past EOF: returns a header with no numbered body
- file too large: returns an error item
- read path is resolved relative to the workspace root if not absolute

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
8. refreshes older live windows for that path in the current partition

Two special non-success results also carry live windows:

- `CONFLICT` if `expected_revision` is stale
- `RANGE_ERROR` if the requested range is invalid for the current file

That lets the model retry immediately against current file state instead of needing an extra `read` round trip.

### Output Shapes

Common function-call output shapes:

| Shape | Meaning |
|---|---|
| `FILE path=... revision=... status=live ...` | Canonical live window |
| `CREATED path=...` | Successful create record |
| `EDITED path=... action=...` | Successful edit record |
| `ALREADY_EXISTS ...` + current view | Create collided with existing file |
| `CONFLICT ...` + current view | Revision mismatch during write |
| `RANGE_ERROR ...` + current view | Invalid line range for current file |
| `FILE path=... status=stale [...]` | Tombstoned live window |
| `ERROR ...` | Validation or filesystem failure with no tracked recovery state |

---

## Request Lifecycle In The Web Harness

### 1. Partition Sync

The harness first resolves the active `ContextPartition` for the request using the normal conversation and compaction machinery.

### 2. Tracked-File Reconciliation

Once the partition is loaded, the harness runs `reconcile_tracked_files(...)`.

This is the point where:

- older live windows pick up external edits
- transient diagnostics normalize back into ordinary live windows
- inaccessible files are tombstoned

### 3. Tool Registration

The web harness then constructs:

```python
file_tool = FileTool(context_partition, workspace_root=Path.cwd())
```

The active partition is passed by reference so same-turn writes can refresh earlier live windows in place before the provider serializes the next round.

### 4. Tool Guidance To The Model

The harness inserts FileTool-specific usage guidance into the system or developer message.

That guidance tells the model to:

- prefer `file_tool` over `shell_command` for normal reads and edits
- treat reads as live windows
- treat write outputs as historical audit trails
- issue writes sequentially
- use `expected_revision` for all line edits
- page through larger files with later `start_line` values

### 5. Same-Turn Refresh

Provider SDKs may batch multiple tool calls in one round. `FileTool` keeps references to pending output items that have been returned but not yet appended to the partition, so a later file-tool call in the same round can still refresh them.

This matters for patterns like:

- read file
- edit same file
- read later window of same file

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

This blocks the final path component from silently changing into a symlink target between validation and open. If that happens, the read or write fails and tracked live windows for that path are tombstoned rather than reading outside the workspace.

### Optimistic Concurrency

Line edits require the caller to supply `expected_revision`.

That protects against stale writes:

- if the file changed since the read, the write does not apply
- instead the tool returns `CONFLICT` plus a fresh current view

This is simpler and safer than trying to merge line edits against unknown intervening changes.

`expected_revision` is only part of the story. The per-path lock plus OS-level `flock` are what make the read-check-write sequence atomic across cooperating same-host writers, so a second request reads after the first commit and returns `CONFLICT` instead of silently overwriting it.

### Per-Path Coordination

There are two coordination layers:

- process-local `asyncio.Lock` keyed by resolved path
- cross-process `fcntl.flock` inside the blocking read/write transaction

The in-process lock prevents same-worker races before operations enter the thread pool. The OS-level advisory lock protects cooperating readers and writers across processes on the same host.

Writers that bypass `file_tool` are outside that contract. Shell commands, editors, and unrelated processes that do not participate in these locks are observed only on the next `file_tool` read for that path or on the next turn's reconciliation pass.

### Text And Newline Semantics

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
| Max rendered lines per read | 200 |
| Max concurrent tracked-path reconciles per turn | 8 |

Files must be readable as UTF-8 text. Binary or oversized files are rejected.

### Crash Atomicity

Writes are protected against cooperating concurrent readers and writers, but they are not crash-atomic. The implementation rewrites files with `seek(0)`, `truncate()`, and `write()` under lock, so a process or host crash mid-write can leave a partial file that the next reconciliation will treat as current on-disk content.

---

## Compaction Integration

`FileTool` is tightly integrated with conversation compaction because live windows are mutable and normal conversation history is otherwise append-only.

### Why Compaction Needs Special Handling

If a live window body were copied into an ancestor summary, the file contents inside that summary could never be refreshed later. The summary would become a fossilized, stale copy of the file.

### Stripping Live Window Bodies From Summary Input

Before summarization, compaction deep-copies the partition and rewrites live-window outputs:

- canonical live windows are replaced with a path-only placeholder
- `ALREADY_EXISTS`, `CONFLICT`, and `RANGE_ERROR` keep their diagnostic headers, but the embedded current-view body is replaced with the same placeholder

Edit records are left intact because they are historical and intentionally frozen.

Example placeholder:

```text
[Live tracked file: /workspace/app.py -- current contents are tracked via the live-window mechanism on subsequent turns, not summarized here.]
```

### Lifting Active Live Windows Into The New Tail

When compaction creates a child partition, it lifts older live windows for any path that is still active in the recency tail.

Activity is keyed by `file_tool.path` annotations, including the frozen edit records produced by successful creates and edits. That lets a recent write keep the relevant older read windows nearby even though the write result itself is not refreshable.

Practical effect:

- if the model recently edited or referenced a file, the next active partition keeps the relevant read windows close to that activity
- the model does not have to rely only on old edit records with shifted line numbers

### Prefix Comparison Ignores Mutable Live-Window Fields

The compaction swap checks whether the cached Redis partition still matches the snapshot prefix. Live-window refreshes are treated specially: mutable fields like `output` and `file_tool.revision` do not count as structural divergence for that comparison.

This prevents a benign live-window refresh from blocking the compaction swap.

### Pending And Committed Child Partitions

Compaction now writes child partitions through explicit `pending` and `committed` states with a `compaction_attempt_uuid`. That is primarily compaction infrastructure, but it matters for FileTool because tracked live windows and edit-record annotations are part of the active partition state that must survive the swap safely.

### Client Relabeling After Compaction

When a compaction swap commits, the compaction-status endpoint can return the child `partition_uuid`. The browser UI rewrites stored message-tree nodes from the old partition UUID to the child UUID so later follow-up turns and session resumes continue from the compacted child rather than an outdated parent pointer.

For deeper compaction details, see [Conversation Compaction](../compaction/README.md).

---

## UI Behavior

The browser UI formats `file_tool` activity entries into concise, user-readable summaries.

Examples:

- `read` -> `Reading /path/to/file from line N`
- `create_file` -> `Creating /path/to/file`
- `replace_lines` -> `Editing /path/to/file lines A-B`
- `insert_lines` -> `Inserting at /path/to/file line N`
- `delete_lines` -> `Deleting /path/to/file lines A-B`

The formatter intentionally hides strict-mode `null` parameters so the activity tray does not show noisy `expected_revision: null` style output for read and create calls.

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
- `tests/unit_tests/test_api_v1_models_annotations.py`
- `tests/unit_tests/test_compaction_file_tool_lift.py`
- `tests/unit_tests/test_compaction_swap.py`
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

Tier A now passes with FileTool enabled in the web harnesses, including Anthropic after the provider-specific tool-schema compatibility fix.

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
| `prokaryotes/tools_v1/file_tool.py` | FileTool implementation, path sandboxing, read/write helpers, reconciliation, rendering |
| `prokaryotes/api_v1/models.py` | `ContextPartitionItem.prokaryotes_annotations`, provider serialization |
| `prokaryotes/openai_v1/web_harness.py` | OpenAI web-harness registration and per-turn reconciliation |
| `prokaryotes/anthropic_v1/web_harness.py` | Anthropic web-harness registration and per-turn reconciliation |
| `prokaryotes/web_v1/__init__.py` | Compaction helpers for stripping, lifting, and live-window-aware prefix comparison |
| `scripts/static/ui.js` | Client-side rendering of `file_tool` activity entries |
| `tests/unit_tests/test_file_tool.py` | Core unit coverage |
| `tests/integration_tests/tier_b/test_file_tool_flow.py` | End-to-end Tier B regression coverage |
| `tests/ui_tests/file_tool_ui.test.js` | UI formatting coverage |
