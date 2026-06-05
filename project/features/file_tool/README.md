# File Tool

## Overview

`FileTool` gives the web harness LLMs structured `read_lines`, `create_file`, `replace_lines`, `insert_lines`, and `delete_lines` actions, with the central invariant that `read_lines` outputs are **live windows**: when the file later changes (model edit, disk edit), the harness refreshes those older windows in place so the next turn sees current file state. Windows for a given resolved path are also kept **deduplicated** — overlapping or contiguous reads consolidate into a non-overlapping cover, so repeated and overlapping reads don't pile up as duplicate windows for the same lines. Edits use line ranges guarded by `expected_revision`, write records stay frozen as an audit trail, and the compactor blanks live windows on its summarization input so file bodies don't fossilize into summaries.

Wired into the Anthropic and OpenAI web harnesses; not yet registered in the script or eval harnesses.

---

## Core Concepts

### Live Windows

A `read_lines` result is a **live window**. It contains:

- the resolved file path
- the current file revision hash
- the requested starting line
- a bounded, line-numbered view of the file

Live windows are `WorkingFileWindow` entries on `Conversation.working_file_windows`. On later turns, the harness may rewrite a window's:

- `rendered_output`
- `revision`
- `view_end_line`
- `line_count`
- `source_kind` (diagnostic source_kinds normalize back to `read_lines` after reconcile)

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

### Window Consolidation

For each resolved path, the windows are kept as a **non-overlapping cover** — reconcile folds any overlap at turn start, and each non-redundant read consolidates the windows its range reaches (read and diagnostic alike) so they no longer overlap. This handles the symlink trio (`CLAUDE.md` / `AGENTS.md` / `README.md` resolving to one path), re-reads of short or over-cap files, and partial-overlap reads — all of which previously minted duplicate windows.

A successful `read_lines` that isn't a coverage hit consolidates: every existing window for the path that **overlaps or touches** the new range (transitively) is merged with it into one interval, and any window disjoint from the new range is left untouched. The interval containing the new call's requested range becomes the **primary** (`window_id = call_id`); if the merged span exceeds `max_lines` it splits, and the leftover intervals become **secondaries** (`window_id = wfw-<uuid>`). The primary and all secondaries share one `origin_call_ids` set — the union of the new call and every absorbed window's origins — and are re-rendered from the freshly-read file text; the absorbed windows are retired. Adjacent ranges the model paged through (`[1, 200]` then `[201, 400]`) merge only when the merged span fits `max_lines`; otherwise the result is two **contiguous** windows — the old one retired and re-minted as the lower secondary.

Reconcile's fold pass re-establishes the cover at turn start. Mid-turn, a write/create can mint a diagnostic window that transiently overlaps a read window; neither a disjoint read nor a coverage-hit read heals it, but the next turn's fold collapses it. Consolidation is greedy and anchored on the new call's start — it deduplicates and keeps windows contiguous, not minimal-cardinality. The pure interval algorithm lives in `intervals.py`.

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

Live-window state lives on `Conversation.working_file_windows` as `WorkingFileWindow` objects:

| Field | Meaning |
|---|---|
| `window_id` | Window identifier, unique within `working_file_windows`. The originating `call_id` for a primary / diagnostic / empty-view window; a fresh `wfw-<uuid>` for a consolidation secondary or a reconcile-fold window. No longer a reliable provenance key once windows merge — use `origin_call_ids`. |
| `path` | Resolved absolute path for the file |
| `status` | `live` or `stale` |
| `revision` | SHA-256 of the current file text |
| `view_start_line` | 1-based first line of the rendered view |
| `view_end_line` | 1-based inclusive last line of the rendered view |
| `requested_end_line` | Required, concrete inclusive end line the window claims to cover (its consolidated boundary). Reconcile re-renders to this fixed extent and never auto-expands the window into a neighbor's range. |
| `line_count` | File's line count at the window's mint/refresh revision. Feeds the coverage check's EOF clamp. |
| `origin_call_ids` | Sorted, deduped, non-empty list of every file-tool `call_id` the window's content traces back to — the new read plus every absorbed prior window. What the compaction and branch/cold-rebuild filters consult. |
| `source_kind` | One of `read_lines`, `already_exists`, `conflict`, `range_error`, `tombstone`. RANGE_TRUNCATED is a response shape only; the window it mints is `read_lines`. |
| `rendered_output` | Cached derived text — what projection emits inside the leading `<working_files>` block |

`WorkingFileWindow` is part of `Conversation` storage and round-trips through Redis and Elasticsearch (as `working_file_windows_json`). Projection emits a leading user-role `<working_files trust="file-content">…</working_files>` block from these windows (see [conversation/README.md — Background-context blocks](../conversation/README.md#background-context-blocks)).

`FileTool` outputs (the `TurnItem`s it returns to the LLM) carry two annotations only:

- `file_tool.path` — the resolved absolute path the call acted on. Branch divergence and cold rebuild read this off kept `TurnExecution.items` to derive active paths.
- `file_tool.persistence` — `"working_file"` for read-like outputs whose durable relevance is on `working_file_windows`, `"history"` for frozen `CREATED` / `EDITED` records. Projection drops historical `function_call_output`s annotated `working_file` (and their paired `function_call`s) on later turns; `history` items ride forward as ordinary transcript.

`REDUNDANT_READ` carries `file_tool.persistence="working_file"` for projection suppression but does **not** allocate a new `WorkingFileWindow` — it points the model at an existing covering window. The redundant call is still recorded as provenance: its `call_id` is appended to the covering window's `origin_call_ids`, so a surviving recency-tail read keeps anchoring a window whose only other origin may be a compacted/pre-tail call.

### Reconciliation

At the start of every web-harness turn, before any `FileTool` call runs, the harness invokes:

```python
await reconcile_working_files(
    conversation.working_file_windows,
    workspace_root=Path.cwd(),
    max_file_bytes=FileTool.max_file_bytes,
    max_lines=FileTool.max_lines,
)
```

That pass, per distinct live path:

- refreshes every live window whose on-disk revision has changed, re-rendering its view and updating `line_count`
- normalizes diagnostic source_kinds (`already_exists`, `conflict`, `range_error`) back to `read_lines` once the underlying file is readable again
- retires obsolete past-EOF placeholders — an empty-view window whose `view_start_line` is now within the file's `line_count`
- **folds** any overlap among a path's live windows into a non-overlapping cover (a no-op when already disjoint), so the per-path invariant holds before any tool call runs
- tombstones windows whose paths are gone or no longer safely readable (`status="stale"`, `source_kind="tombstone"`)

This is what lets the next turn see current file state even when the previous turn did not itself perform the edit, and what guarantees the `REDUNDANT_READ` coverage check inside `FileTool` runs against post-reconcile state.

The portability contract: any harness that consumes `Conversation.working_file_windows` for current file context must run `reconcile_working_files(...)` at turn start. The web harness already does this; future harnesses (Slack, etc.) follow the same pattern.

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

`read_lines` reads disk **first**, refreshes every live window for the path against that fresh content, and only then checks coverage — so the decision always runs against current file state, correct even when the file grew or shrank mid-turn. A request is **covered** when an existing window starts at or before the request and a fresh disk read would return no lines past the window's `view_end_line` (the fresh read's effective end is clamped by the requested end, by `line_count` for EOF, and by the per-call cap `start_line + FileTool.max_lines - 1`). This single extent check unifies the old exact-match / subset / EOF / cap cases. A covered request returns a short `REDUNDANT_READ` diagnostic — one of three variants (EOF / per-call cap / subset) depending on which clamp made coverage hold, with paging guidance only where paging would actually surface unread lines — and mints no window.

Coverage eligibility is narrow:

- only `read_lines` windows count (RANGE_TRUNCATED mints a `read_lines` window; the transient `already_exists` / `conflict` / `range_error` diagnostics and `tombstone`s do not)
- a window that was a **diagnostic before this read's refresh** is excluded — its id is snapshotted pre-refresh, because the refresh normalizes it to `read_lines`; a pre-existing diagnostic must fold through consolidation (recording this call's provenance), not short-circuit
- the window's current content must have actually been **shown to the model this turn** (the exposure gate, below)

A coverage miss falls through to consolidation — except an empty file or a past-EOF read, which short-circuits to an empty-view placeholder first (see Behavior details). Consolidation may still retire and merge the reached windows.

#### Exposure gate

The leading `<working_files>` projection is built once at turn start and never rebuilt mid-turn, and a `REDUNDANT_READ` is only a pointer — so a window whose content the refresh silently advanced to the current revision but never *re-rendered* to the model (a sibling refreshed by a disjoint read, or an in-place external change between two reads) must not grant `REDUNDANT_READ`, or the model would act on content it was never shown. `FileTool` keeps a turn-local `window_id → shown-revision` map, seeded at construction from the reconciled (about-to-be-projected) windows and updated only by model-facing renders — never by refresh; coverage requires `shown-revision == window.revision`. The cost is bounded over-conservatism: a same-turn re-read of a consolidation-extended primary or a split-off secondary (neither rendered whole) re-renders instead of deduping, and self-heals next turn when projection re-seeds the map.

Behavior details:

- empty file: returns `FILE ... status=live line_count=0`; the coverage-miss path retires prior windows and mints a `(1, 0)` placeholder
- start line past EOF on a non-empty file: mints an empty-view placeholder `(start_line, start_line - 1)` and leaves other windows untouched
- windows are **fixed-extent**: every window stores a concrete `requested_end_line` (its consolidated boundary), so reconcile re-renders to that boundary and never auto-expands on file growth. To see beyond a window the model issues a new `read_lines` that triggers a fresh consolidation. (This drops the old open-ended auto-expand-on-growth behavior; a separate `refresh_intent` field is the path back if it's ever needed.)
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
| `RANGE_TRUNCATED ...` + current view | `read_lines` exact span exceeded the per-call cap and content was clipped; window covers the first `FileTool.max_lines` lines with paging guidance for the remainder. Response shape only — the minted window is `source_kind="read_lines"` and consolidates like any other read. |
| `REDUNDANT_READ ...` | `read_lines` requested span was already fully covered by an existing live window; diagnostic points at that window instead of re-rendering. Three variants (EOF / per-call cap / subset). Fires more often under the unified extent check; mints no window but appends the call to the covering window's `origin_call_ids`. |
| `FILE path=... status=stale [...]` | Tombstoned live window |
| `ERROR ...` | Validation or filesystem failure with no tracked recovery state |

---

## Request Lifecycle in the Web Harness

The harness resolves the active `Conversation` snapshot for the request (see [conversation/README.md](../conversation/README.md)). `working_file_windows` is a first-class field on the snapshot; FileTool reads and mutates that list directly.

### 1. Working-File Reconciliation

Before any `FileTool` call runs in the turn, the harness runs:

```python
await reconcile_working_files(
    conversation.working_file_windows,
    workspace_root=Path.cwd(),
    max_file_bytes=FileTool.max_file_bytes,
    max_lines=FileTool.max_lines,
)
```

This is the point where:

- older live windows pick up external edits (revision changed → re-render, `line_count` updated)
- transient diagnostic source_kinds (`already_exists`, `conflict`, `range_error`) normalize back to `read_lines` against fresh on-disk text
- overlapping windows for a path **fold** into a non-overlapping cover, and obsolete past-EOF placeholders are retired
- inaccessible files are tombstoned (`status="stale"`, `source_kind="tombstone"`)

### 2. Tool Registration

The web harness then constructs:

```python
file_tool = FileTool(
    working_file_provider=lambda: conversation.working_file_windows,
    workspace_root=Path.cwd(),
)
```

The `working_file_provider` is called by `FileTool` on every action so same-turn calls see prior windows minted earlier in the same turn — the tool mutates the list in place, no `_pending_result_items` bridge needed. Constructing `FileTool` here — **after** reconcile, **before** projection — is also what makes the exposure gate correct: it seeds its turn-local `window_id → shown-revision` map from the reconciled windows that are about to be projected (see [Exposure gate](#exposure-gate)).

### 3. Tool Guidance to the Model

The harness inserts FileTool-specific usage guidance into the system or developer message.

That guidance tells the model to:

- prefer `file_tool` over `shell_command` for normal reads and edits
- treat `read_lines` outputs as live windows
- treat write outputs as historical audit trails
- issue writes sequentially
- use `expected_revision` for all line edits
- page through larger files with later `start_line` values

### 4. Same-Turn Refresh

Provider SDKs may batch multiple tool calls in one round. Because `working_file_provider` returns the live `conversation.working_file_windows` list and `FileTool` mutates it in place, a later file-tool call in the same round sees windows minted by earlier calls — covering patterns like read → edit → read-another-range within one model turn.

The durable file context is re-projected each turn as the leading `<working_files>` block (historical `read_lines` `function_call_output`s are filtered out — see [Tracked Metadata](#tracked-metadata)), so provider-side prompt caches can lose reuse when a tracked file changes: the block's content shifts and invalidates the cached prefix from that point on. In practice that means editing a tracked file may turn the next turn back into fresh input-token billing until a new stable prefix forms. Unchanged files do not churn cache state because windows are only re-rendered when the file revision changes or a transient diagnostic normalizes back into an ordinary `FILE ...` view.

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

Writers that bypass `file_tool` are outside that contract. Shell commands, editors, and unrelated processes that do not participate in these locks are observed on the next `read_lines` call for that path — which always reads disk and refreshes the path's windows before checking coverage, so even a span an existing window covers renders the changed content rather than hiding it behind `REDUNDANT_READ` (the exposure gate keeps a silently-refreshed window from short-circuiting) — or on the next turn's reconciliation pass.

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

Files must be readable as UTF-8 text. Binary or oversized files are rejected.

### Crash Atomicity

Writes are protected against cooperating concurrent readers and writers, but they are not crash-atomic. The implementation rewrites files with `seek(0)`, `truncate()`, and `write()` under lock, so a process or host crash mid-write can leave a partial file that the next reconciliation will treat as current on-disk content.

---

## Compaction Integration

Live windows are mutable; if their bodies fossilized into a summary, the summary would carry a stale copy of the file that could never be refreshed. Two mechanisms prevent this.

### Keeping File Bodies Out of the Summary Input

`HarnessBase._summarize_and_compact` projects `pre_tail_conv` with `working_file_windows=[]` and `ancestor_summaries=[]`, so the summarizer's input carries no leading `<working_files>` block at all. Historical `function_call_output`s annotated `file_tool.persistence="working_file"` (and their paired `function_call`s) are also dropped from projection on later turns, so no live-window bodies reach the summarizer through the transcript either. Frozen edit records (`persistence="history"`) and non-file-tool tool history ride through unchanged.

### Carrying Working Files into the New Tail

When compaction creates a child `Conversation` snapshot, `_cas_swap_child` carries `working_file_windows` forward from the **live Redis snapshot at CAS time** (`current.working_file_windows`, not the deep-copy snapshot taken at compaction start), via `_carry_forward_windows`. The filter is keyed on `origin_call_ids`, not `window_id` — consolidation and fold can merge several calls (and synthetic `wfw-*` ids) into one window, so a single id no longer identifies provenance. **Keep a window iff at least one of its `origin_call_ids` escapes the pre-tail span:**

```python
pre_tail_call_ids = _file_tool_call_ids_in(prep.pre_tail_turns)
carried_windows = _carry_forward_windows(current.working_file_windows, pre_tail_call_ids)
# keeps w iff set(w.origin_call_ids) - pre_tail_call_ids is non-empty
```

A **surviving origin** is any origin not in the pre-tail: a recency-tail call, a post-snapshot call finalized during in-flight summarization (race-safe — `pre_tail_turns` is loaded in `_prepare_compaction` well before CAS and the filter never reads post-snapshot turns), or a compacted-ancestor call that appears in no current turn at all (carryforward). A window is dropped only when **every** origin is a pre-tail call.

The behavioral tradeoff is narrower than before. A recency-tail call now anchors a pre-tail-minted window whenever it adds provenance — a re-read that consolidates the window, or a `REDUNDANT_READ` whose `call_id` is appended to the covering window's origins. A bare edit refreshes content without touching `origin_call_ids`, so it does not anchor; if every origin stays pre-tail the window is dropped and the model can re-read. A merged window can carry a pre-tail origin alongside a surviving one and is kept whole — accepted benign over-retention (the window re-renders against live disk content every reconcile, so the carried lines are current, not stale branch state).

### Branch Divergence

Case A branch divergence (and cold rebuild) apply a two-gate filter to `working_file_windows`: keep a window if its path is active in the kept turns AND any of its `origin_call_ids` is a **surviving origin** — `any(o ∈ kept_call_ids)` OR `any(o ∉ source_call_ids)` (the carryforward bucket: an origin in no source-reachable `TurnExecution` came from a compacted ancestor). The second disjunct is `any(o ∉ source_call_ids)`, **not** `not any(o ∈ source_call_ids)` — a merged window carrying both an ancestor-carryforward origin and a discarded-sibling origin is kept on the surviving ancestor. A window is dropped only when every origin is a discarded source origin (in `source_call_ids`, not `kept_call_ids`). Active paths come from `file_tool.path` annotations on the kept turns' file-tool outputs, which cover every call shape including edits and REDUNDANT_READs. See [conversation/README.md — Branches and Snapshots](../conversation/README.md#branches-and-snapshots).

### Client Relabeling After Compaction

When a compaction swap commits, the compaction-status endpoint can return the child `snapshot_uuid`. The browser UI rewrites stored message-tree nodes from the old snapshot UUID to the child UUID so later follow-up turns and session resumes continue from the compacted child rather than an outdated parent pointer.

For deeper compaction details, see [Conversation Compaction](../compaction/README.md).

---

## UI Behavior

The browser UI formats `file_tool` activity entries into concise summaries (`Reading /path from line N`, `Editing /path lines A-B`, etc.) attached to assistant nodes. The formatter hides strict-mode `null` parameters so the activity tray doesn't show noisy `expected_revision: null` output for `read_lines` and create calls.

---

## Testing

Unit coverage lives in `tests/unit_tests/test_file_tool*.py` — including `test_file_tool_window_dedup.py`, which covers interval consolidation, the unified coverage check and exposure gate, the reconcile fold, and the generalized compaction / branch-divergence origin filters (it absorbed the old `test_compaction_pre_tail_filter.py` and `test_origin_filter.py`, which were removed) — plus `test_working_file_models.py` and `test_reconcile_working_files.py`. Tier B end-to-end coverage is in `tests/integration_tests/tier_b/test_file_tool_flow.py`; UI-side formatting is in `tests/ui_tests/file_tool_ui.test.js`. Tier A passes with FileTool enabled on both Anthropic and OpenAI web harnesses.

---

## Known Limitations

- Text only. The tool assumes UTF-8 and does not support binary patching.
- No rename/move/chmod operations. It is intentionally scoped to reads and line edits.
- Large files require paging or fallback tooling. Only 200 lines are rendered per view and files over 1 MB are rejected.
- Script and eval harnesses do not currently expose FileTool.
- A `WorkingFileWindow` carried forward through compaction can lag one refresh cycle if the file changed between compaction CAS and the next turn. The next turn's `reconcile_working_files` repairs it.
- Consolidation deduplicates overlapping/contiguous windows but does not cap the **number** of windows: a model that reads many disjoint ranges of a long file still accumulates one window per disjoint range. A per-path window-count cap is deferred.
- Cleanup for stale `pending` compaction children is deferred to later recovery/sweeper work. That is not a FileTool correctness gap, but it is related infrastructure.

---

## Relevant Code Files

| File | Role |
|---|---|
| `prokaryotes/tools_v1/file_tool/__init__.py` | `FileTool` class. Read/create/edit dispatch, the unified coverage check + turn-local exposure map, interval consolidation of windows (via `intervals.py`), `WorkingFileWindow` minting and in-place refresh; delegates locked file I/O to `reads.py`, refresh/fold/tombstone helpers to `live_windows.py`, path sandboxing to `paths.py`. |
| `prokaryotes/tools_v1/file_tool/intervals.py` | Pure interval consolidation — `Interval`, `ConsolidationResult`, `consolidate_intervals`. No I/O; computes the post-read non-overlapping cover for one path. |
| `prokaryotes/tools_v1/file_tool/live_windows.py` | `refresh_windows_for_path`, `fold_windows_for_path` (turn-start overlap fold), `tombstone_windows_for_path`, `reconcile_working_files` — operate on `Conversation.working_file_windows`. |
| `prokaryotes/tools_v1/file_tool/paths.py` | Workspace sandboxing — `_resolve_path`, `_open_text_file_no_follow`, `_raise_if_file_too_large`, `FileToolFileTooLargeError`. |
| `prokaryotes/tools_v1/file_tool/reads.py` | Locked-read primitives (`_locked_read_text`, `_read_text_under_file_tool_lock`) and the shared per-path `asyncio.Lock` registry. |
| `prokaryotes/tools_v1/file_tool/rendering.py` | `render_create_record`, `render_edit_record`, `render_live_window`, `render_tombstone`, `render_view`, line-edit text mutation, and the `CURRENT_VIEW_MARKER_PREFIX` constant. |
| `prokaryotes/tools_v1/file_tool/validation.py` | Payload field validation for `read_lines`, `create_file`, and the write actions. |
| `prokaryotes/conversation_v1/models.py` | `WorkingFileWindow` (required `line_count` and `origin_call_ids`, concrete `requested_end_line`, `origin_call_ids` validator), `Conversation.working_file_windows`, `Conversation.working_files_block()`, `coverage_eligible()`. |
| `prokaryotes/conversation_v1/project.py` | `project_for_llm` — emits the leading `<working_files>` block and filters historical `file_tool.persistence="working_file"` outputs (and their paired calls). |
| `prokaryotes/harness_v1/web.py` | Web-harness `FileTool` registration, `working_file_provider` wiring (`lambda: conversation.working_file_windows`), and turn-start `reconcile_working_files`. |
| `prokaryotes/context_v1/compaction.py` | `_file_tool_call_ids_in` and `_carry_forward_windows` — the pre-tail origin filter `_cas_swap_child` uses to carry `working_file_windows` forward to the child snapshot. |
| `prokaryotes/context_v1/conversation_sync.py` | `_active_paths_in_turns`, `_filter_windows_by_active_path_and_origin` — Case A divergence and cold-rebuild two-gate filter on `working_file_windows`. |
| `prokaryotes/search_v1/conversations.py` | `working_file_windows_json` persistence; `conversation_from_doc` / `working_file_windows_from_doc`. |
| `scripts/static/ui.js` | Client-side rendering of `file_tool` activity entries |
| `tests/unit_tests/test_file_tool.py` | Core unit coverage |
| `tests/integration_tests/tier_b/test_file_tool_flow.py` | End-to-end Tier B regression coverage |
| `tests/ui_tests/file_tool_ui.test.js` | UI formatting coverage |
