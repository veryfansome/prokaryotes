# File Tool with Tracked Context Synchronization

## Goals

- Add a dedicated file utility that fits the existing `FunctionToolCallback` pattern.
- Make file reads and edits more efficient than the current shell-command-based workflow.
- Provide line-numbered file views so the model can reason about and modify specific ranges.
- Reduce stale file content in model context by introducing a harness-managed canonical file state per tracked path.
- Support external on-disk file changes between turns without silently continuing from obsolete snapshots.

## Non-goals

- Replacing `shell_command` as a general-purpose workspace tool.
- Changing the client-server conversation API protocol.
- Solving binary file editing or large-repo indexing in this feature.
- Implementing live filesystem watching; synchronization occurs during request assembly and file-tool operations.

## Current State

### Observed implementation facts

- `ShellCommandTool` is a reusable `FunctionToolCallback` that accepts JSON arguments, runs `asyncio.create_subprocess_shell(...)`, and returns a `ContextPartitionItem` of type `function_call_output` whose `output` is plain text containing exit code, stdout, and stderr.
- `FunctionToolCallback` currently exposes a single `tool_spec`, `system_message_parts`, `name`, and `async call(arguments: str, call_id: str) -> ContextPartitionItem | None`.
- `WebHarness.post_chat(...)` creates a `ShellCommandTool` and `ThinkTool`, injects each tool's `system_message_parts` into the developer message, then calls `llm_client.stream_turn(..., tool_callbacks=tool_callbacks)`.
- `ContextPartitionItem` has generic fields for messages, function calls, and function call outputs. It does not currently include structured metadata fields for file snapshots.
- `WebBase.finalize(...)` persists the full `ContextPartition` in Redis and search storage after removing the injected developer message.
- `WebBase` already supports server-side partition mutation during compaction by creating a new `ContextPartition` with ancestor summaries and a reduced tail, then swapping it into Redis and search storage.
- The current client conversation payload only carries user/assistant messages; tool-call history is server-side state reconstructed from persisted `ContextPartition` items.

### Problem statement

Today, files are typically inspected and edited through `shell_command`, which causes several issues:

- repeated file reads inject duplicate content into the context window;
- small edits often require whole-file rewrites or brittle shell one-liners;
- old file snapshots remain in persisted tool outputs after the file changes on disk or through subsequent tool calls;
- compaction currently has no file-specific notion of canonical state versus stale snapshots.

## Core Concepts and New Abstractions

### 1. `FileTool`

Add a new reusable `FunctionToolCallback` implementation, tentatively `prokaryotes/tools_v1/file_tool.py`, that exposes structured file operations through a single tool.

`FileTool` takes `context_partition` as a constructor argument and is instantiated per-request in `post_chat()` after `sync_context_partition()` returns the authoritative partition, consistent with `ThinkTool` taking `llm_client` and `model` at construction time. The rationale is in Write behavior below.

Both LLM clients dispatch tool callbacks within a round concurrently (`asyncio.create_task` + `asyncio.gather`). `FileTool` therefore holds two layers of locking:

- a per-instance `asyncio.Lock` that serializes its own `call()` invocations within one request — guards in-memory partition mutation and prevents concurrent refresh-in-place loops from interleaving;
- a class-level per-path `asyncio.Lock` keyed by resolved absolute path, plus an OS-level `fcntl.flock(LOCK_EX)` taken on the target file's own descriptor inside the threaded write transaction. Together these make the read → revision-check → write sequence atomic across cooperating `FileTool` writers on the same host. Without them, two requests could both pass an `expected_revision == A` check on stale-but-still-current-at-read-time content and silently overwrite each other; with them, the second writer reads after the first writes and returns a conflict instead.

External writers that bypass `file_tool` (shell commands, editors, other processes that don't use `flock`) are not coordinated by these locks; their changes are detected on the next read or reconciliation rather than at write time. All filesystem I/O uses `asyncio.to_thread` to avoid blocking the event loop.

The tool exposes a single `action` enum parameter. Proposed values:

- `read` — return a line-numbered view from `start_line` up to `max_lines` lines.
- `replace_lines` — replace an inclusive line range with new text.
- `insert_lines` — insert text at a given line boundary.
- `delete_lines` — delete an inclusive line range.

### 2. File tracking via existing partition items

Rather than maintaining a separate tracked-file registry, the harness derives which files are currently tracked by scanning `context_partition.items` for `function_call_output` items whose `prokaryotes_annotations` carry `"file_tool.status": "live"`. Each live item IS a tracking record — it carries the current `file_tool.revision`, `file_tool.view_start_line`, and `file_tool.view_end_line` annotations directly.

Multiple live items per path can co-exist: e.g., the model reads lines 1–200 and then lines 201–400 of the same file. Both items remain live and represent independently useful views into the current file.

A live item is a **live window**, not a static snapshot. The framing borrows from human vision: rather than a flipbook of disjoint snapshots, perception is a single continuously-updated representation of a perspective. Live windows play the same role for the model — they are not records of what the file looked like at the time of the read, but views of what the file looks like *now*. Whenever the file changes — whether through a subsequent `file_tool` write or an external edit detected at reconciliation time — every live window for that path is refreshed in-place: its stored view range is re-rendered against the new content, and its revision and end-line annotation are updated. The model therefore sees earlier `function_call_output` items in its conversation history change between turns to track the current file state. This is the central departure from the default assumption that tool outputs are immutable, and the system message guidance below makes it explicit to the model.

Compaction does require explicit handling so live windows for actively-worked files survive: the recency tail's `file_tool.path` annotations identify "active paths", and pre-tail live windows for those paths are lifted into the tail at compaction time. See "Compaction behavior" for the full mechanism.

### 3. Canonical file output protocol

`FileTool` produces three distinct kinds of `function_call_output` text. Each kind has one job; the harness identifies them by their leading marker and (where applicable) by `prokaryotes_annotations`.

**Live window** (read result, conflict result, refreshed earlier window). Carries `prokaryotes_annotations` with `file_tool.status="live"` so it is picked up by the live-window scan and refreshed by future changes.

```text
FILE path=project/wip/README.md revision=abc123 status=live lines=1-40 line_count=87
1 | # Feature Planning and Implementation Process
2 | ...
```

**Edit record** (write result). A frozen historical record of what changed. If live windows are the model's perception of files, edit records are its episodic memory: what the model *did*, distinct from what the model currently perceives. Carries `prokaryotes_annotations = {"file_tool.path": str(path)}` — just the path, with no `file_tool.status`, so it is *not* a live window, is never refreshed, and survives across turns so the model can retrace its edits or revert them. The path annotation is what compaction uses to identify "active paths" worth preserving live windows for (see "Compaction behavior" below). Removed-line numbers reference the file before the edit; added-line numbers reference the file after.

```text
EDITED path=project/wip/README.md action=replace_lines
revision: abc123 → def456
line_count: 87 → 89

Removed (lines 10-15):
10 | old a
11 | old b
12 | old c
13 | old d
14 | old e
15 | old f

Added (lines 10-17):
10 | new a
11 | new b
12 | new c
13 | new d
14 | new e
15 | new f
16 | new g
17 | new h
```

`delete_lines` produces only a `Removed` block; `insert_lines` produces only an `Added` block. If either side of the diff exceeds `max_lines`, that side is truncated with a `... <n> more lines truncated ...` marker so the record stays bounded.

**Tombstone** (file became inaccessible between turns). Replaces a live window's `output` in-place; sets `file_tool.status="stale"`.

```text
FILE path=project/wip/README.md status=stale [no longer accessible: FileNotFoundError]
```

Stale markers exist only for the tombstone case. Write-superseded windows are never marked stale — they are refreshed in-place against the new file content (see "Write behavior").

The exact wire format may be text-only for readability, but it should be generated from structured metadata rather than ad hoc strings.

### 4. Harness-managed tracked-file reconciliation

Before each model turn, the harness scans `context_partition.items` for live windows, checks each against the filesystem, and refreshes in-place any whose on-disk content has changed since the window was last rendered.

This is stronger than a pure append-only protocol because the persisted tool history is owned by the harness, not the client.

## Data Model Changes

### `ContextPartition`

No new fields required. Tracking state is derived entirely from existing `items`.

### `ContextPartitionItem`

Add a single optional field rather than individual typed fields:

```python
prokaryotes_annotations: dict[str, str] | None = None
```

This follows the Kubernetes annotations pattern: different harness components can attach arbitrary string key-value metadata to an item without requiring schema changes each time. Keys are dot-namespaced by component to avoid collisions.

`FileTool` uses the following keys:

| Key | Values |
|---|---|
| `file_tool.path` | absolute resolved path |
| `file_tool.revision` | `sha256(content).hexdigest()` |
| `file_tool.status` | `"live"` or `"stale"` |
| `file_tool.view_start_line` | stringified int |
| `file_tool.view_end_line` | stringified int |

These annotations are only populated on file tool `function_call_output` items. `FileTool` is responsible for parsing its own int-valued annotations back from strings when it reads them.

Reasoning:

- the harness can mutate or compact stale file outputs reliably without brittle string parsing;
- future tools or summarization passes can attach their own metadata without modifying `ContextPartitionItem`.

**Important**: `to_openai_input()` uses `item.model_dump(exclude_none=True, ...)`, which would include `prokaryotes_annotations` in the dict sent to the OpenAI Responses API. It must be excluded: after the `text_preamble` removal below, this becomes `exclude={"prokaryotes_annotations"}`. `to_anthropic_messages()` is not affected because it constructs its output from explicit field accesses rather than `model_dump()`.

### Removal: `text_preamble`

In addition to adding `prokaryotes_annotations`, drop the existing `ContextPartitionItem.text_preamble` field.

`text_preamble` currently captures text the model emits before a tool call within a tool-use round, and is replayed on subsequent turns by `to_anthropic_messages()` (as a text block in the same assistant message as the `tool_use`) and by `to_openai_input()` (as a synthesized preceding `assistant` message item). It is distinct from the streaming `progress_message` event the harness already emits — `progress_message` displays the same text transiently in the UI but is never persisted in conversation state.

Pre-tool-call narration in practice is procedural ("Now I'll read...", "Let me check...") rather than chain-of-thought that subsequent turns depend on; reasoning models carry the actual reasoning in the providers' separate reasoning traces. The streaming UX is unaffected because `progress_message` is independent.

This intersects with "Compaction behavior" below: that section lifts pre-tail `(function_call, function_call_output)` pairs into the tail to preserve live windows for actively-worked paths. With `text_preamble` in place, lifted `function_call` items would carry narration that was emitted in their original round and surface it in their new lifted position, claiming the model said something it never said in that round. Removing `text_preamble` first eliminates this corner case entirely.

Affected sites:

- `prokaryotes/api_v1/models.py` — remove the field; remove the `text_preamble`-handling branch in `to_anthropic_messages()` (the conditional text-block prepend) and the synthetic-assistant-message branch in `to_openai_input()` (`if item.type == "function_call" and item.text_preamble: ...`); simplify the `model_dump` exclude to just `{"prokaryotes_annotations"}`.
- `prokaryotes/openai_v1/__init__.py` — remove the `tool_preamble` accumulator and the `text_preamble` assignment in `handle_response_stream_event()`. `round_text` continues to drive `progress_message`.
- `prokaryotes/anthropic_v1/__init__.py` — remove the `text_preamble=...` argument when constructing `function_call` items in `stream_turn()`. `round_output` continues to drive `progress_message`.

## Protocol Changes

### Tool schema

Add a new tool named `file_tool` with a single function schema driven by an `action` enum.

Example shape:

```json
{
  "action": "replace_lines",
  "path": "prokaryotes/openai_v1/web_harness.py",
  "start_line": 10,
  "end_line": 18,
  "new_text": "...",
  "expected_revision": "abc123"
}
```

All parameters are required (enforced by `strict=True` on `ToolSpec`). Parameters that do not apply to a given action are passed as `null`. Nullable types are expressed in the JSON schema as `{"type": ["<type>", "null"]}`.

| Parameter | Type | Pass `null` when |
|---|---|---|
| `action` | `string` | never |
| `path` | `string` | never |
| `expected_revision` | `string \| null` | `read` |
| `start_line` | `integer \| null` | reading from the beginning of the file |
| `end_line` | `integer \| null` | `read`, `insert_lines` |
| `new_text` | `string \| null` | `read`, `delete_lines` |

`FileTool` exposes a `max_lines` class constant (similar to `ShellCommandTool.max_output_lines`) that caps how many lines a read returns. The model does not choose this value per call.

### Developer message guidance

`FileTool.system_message_parts` should instruct the model to:

- use `file_tool` instead of `shell_command` for routine file reads/edits;
- prefer targeted line-range operations over whole-file rewrites;
- treat each `file_tool` read output as a **live window** into the file, not a static snapshot. The harness keeps every prior read window for a file in sync with the current on-disk content: when a subsequent write or an external edit changes the file, the harness updates earlier `function_call_output` items in-place so their rendered views and revisions reflect the current file content. Earlier windows in the conversation history are therefore authoritative for what the file looks like *now*, not what it looked like at the time of the read;
- treat each `file_tool` write output (an **edit record**) as a frozen historical audit trail, not a current view. The line numbers in an edit record's `Removed` and `Added` blocks refer to file state at the time of *that* edit. After subsequent edits to the same file, those absolute line numbers may have shifted and no longer point to the same content. To target the same content for further edits, always consult the most recent live window for the path; never carry line numbers forward from an edit record;
- emit file edits sequentially rather than in parallel — issue one `file_tool` write at a time and wait for its result before issuing the next, especially when multiple edits target the same file. Concurrent writes against the same path will produce a conflict on all but one of them, which the model will then have to recover from.

### UI rendering

`ui.js` dispatches `tool_call` events to per-tool formatter functions (`formatThinkToolCallMarkdown`, `formatShellCommandToolCallMarkdown`, etc.). Without a dedicated handler, `file_tool` calls fall through to `formatGenericToolCallMarkdown`, which dumps all parameters including the nulls required by strict mode — not useful.

Add `formatFileToolCallMarkdown` to `ui.js` and register it in `formatToolCallMarkdown`. The formatter should be action-aware and suppress null parameters:

- **`read`**: `Reading \`<path>\` from line <start>` (or `Reading \`<path>\`` if `start_line` is null)
- **`replace_lines`**: `Editing \`<path>\` lines <start>–<end>` followed by `new_text` in a fenced code block
- **`delete_lines`**: `Deleting \`<path>\` lines <start>–<end>`
- **`insert_lines`**: `Inserting at \`<path>\` line <start>` followed by `new_text` in a fenced code block

No other client protocol changes are required.

## Detailed Behavior

### Read behavior

1. Resolve `path` against `workspace_root` and verify it does not escape it.
2. Read file contents from disk using UTF-8 encoding via `asyncio.to_thread`.
3. Compute `revision` as `sha256(text.encode("utf-8")).hexdigest()`.
4. Build a line-numbered view from `start_line` (default 1 if null) up to `start_line + max_lines - 1`, capped at the end of the file.
5. Return a `ContextPartitionItem` of type `function_call_output` containing the readable file view and `prokaryotes_annotations` populated with `file_tool.path`, `file_tool.revision`, `file_tool.status="live"`, `file_tool.view_start_line`, and `file_tool.view_end_line`.

### Write behavior

`FileTool` holds a reference to `context_partition` (injected at construction time) so it can refresh prior live items for the path directly from inside `call()`.

`expected_revision` is required for all write operations (`replace_lines`, `insert_lines`, `delete_lines`). The system message guidance should instruct the model to always obtain a revision from a preceding `read` call and pass it on writes.

1. Resolve `path` against `workspace_root` and verify it does not escape it. Reject the call with an error result if `expected_revision` is missing.
2. Acquire the class-level per-path `asyncio.Lock` (`FileTool._get_path_lock(str(path))`). This serializes same-process writers before they enter the thread pool.
3. In `asyncio.to_thread`, run the locked write transaction. The transaction returns `(result_item, current_text, current_revision)` describing both the result the caller will return and the file's post-transaction on-disk state, which the caller uses to refresh prior live windows.
   1. Open the target file with `open(path, "r+", encoding="utf-8")` and acquire `fcntl.flock(LOCK_EX)` on its descriptor. This is the durable layer that survives multi-process worker setups.
   2. Read current contents through that descriptor and compute `current_revision = sha256(text.encode("utf-8")).hexdigest()`.
   3. If `expected_revision` does not match `current_revision`, build a conflict result from the just-read content (see "Conflict and edge cases") and return `(item, original_text, current_revision)` without writing. The advisory lock releases when the descriptor closes.
   4. If the requested line range is out of bounds, build a range-error result the same way and return `(item, original_text, current_revision)` without writing.
   5. Otherwise apply the line-based mutation, `seek(0)` + `truncate()` + `write(updated_text)` through the same descriptor (so the write happens under the same flock that protected the read), and compute `new_revision`.
   6. Build the edit record (see "Canonical file output protocol") and return `(item, updated_text, new_revision)`. Closing the descriptor on `with` exit flushes the write and releases the advisory lock.
4. Release the per-path `asyncio.Lock` once the threaded transaction returns.
5. Refresh all prior live items for this path in-place against `(current_text, current_revision)` using `_refresh_live_windows`: re-render each window from its stored `file_tool.view_start_line` against the new content, replace `output` with the fresh rendered view, and update `file_tool.revision` and `file_tool.view_end_line` in the annotations. The items remain `"live"`. Refresh runs for all three result kinds — successful edits, conflicts, and range errors — so older windows for this path always end up at the same revision the just-built result item reports. It is a no-op for items already at the current revision (the common case after a range error). Refresh runs outside both locks because it touches only this request's in-memory partition.
6. Return the result item to the caller (an edit record on success, or the conflict / range-error item built in step 3). Edit records carry `prokaryotes_annotations = {"file_tool.path": str(path)}` — just the path, no `file_tool.status` — so they are not live windows and are never refreshed; they serve as the model's audit trail so it can retrace or undo what it just did even after live windows have moved on, and the path annotation lets compaction recognize this path as active.

Conflict and range-error results carry the same `prokaryotes_annotations` shape as a successful read for the just-loaded content, so they double as a fresh live window the model can read line numbers from when retrying.

### Reconciliation before each turn

Add a module-level `reconcile_tracked_files()` function in `prokaryotes/tools_v1/file_tool.py` (alongside `FileTool`). Each harness's `post_chat()` imports and calls it after `sync_context_partition()` returns and before the developer message is assembled. Placing the function in `tools_v1/file_tool.py` rather than on `WebBase` keeps it in the same module that defines the `file_tool.*` annotation keys it operates on, and avoids a `web_v1 → tools_v1` dependency that would otherwise be needed.

Each live file item already has a matching `function_call` item in the partition, so updating `item.output` and annotations in-place is valid — no orphaned items are created.

1. Scan `context_partition.items` for `function_call_output` items whose `prokaryotes_annotations` carry `"file_tool.status": "live"`, grouping all live items by `file_tool.path`.
2. For each path, attempt to read the file once from disk with UTF-8 encoding and compute current revision as `sha256(text.encode("utf-8")).hexdigest()`. Use `asyncio.to_thread` for the read.
3. If the read raises `FileNotFoundError`, `IsADirectoryError`, `PermissionError`, or `UnicodeDecodeError`, mark every live item for that path stale in-place: set `file_tool.status = "stale"` and replace `item.output` with a tombstone marker that names the failure mode (e.g. `"FILE path=<p> status=stale [no longer accessible: <reason>]"`). Continue to the next path.
4. For each live item belonging to a successfully-read path:
   - if the current revision matches `item.prokaryotes_annotations["file_tool.revision"]`, leave the item alone;
   - otherwise update the item in-place: re-render the file view from `file_tool.view_start_line` up to `FileTool.max_lines` lines, replace `item.output` with the fresh rendered view, and update `file_tool.revision` and `file_tool.view_end_line` in the annotations. The item remains `"live"`.

#### Contract for partition consumers

`reconcile_tracked_files` is the single canonical mechanism for bringing live windows back to current on-disk state. The persistence layer (Redis, Elasticsearch) preserves whatever live-window state was in the partition at the moment it was written, which may already be one or more turns stale by the time it is read back.

Any code path that loads a `ContextPartition` from Redis or ES and cares about current file state must call `reconcile_tracked_files(partition)` before consuming live windows. Today the two web harnesses (`anthropic_v1/web_harness.py`, `openai_v1/web_harness.py`) already do this in `post_chat()` after `sync_context_partition()`. Future consumers — additional harnesses, search-result rendering, debug tooling, alternative compaction strategies — must adhere to the same contract or accept the displayed file state may lag the disk.

This contract is what makes the modulo-live-window prefix check (see "Compaction behavior") safe: a swap may install lifted windows that briefly trail the on-disk state, but the next `post_chat()` repairs them before the model sees them.

### Compaction behavior

Live windows must survive compaction for files the model is still actively working on, otherwise the live-window framing breaks down: a file read at turn 1 would silently lose its tracking on the next compaction even if the model is mid-edit at turn 30. The compaction logic uses the recency tail as a signal of which paths are active and lifts pre-tail live windows for those paths into the tail.

1. Compute the recency tail as today.
2. Scan the recency tail for any item carrying a `file_tool.path` annotation. This includes both live windows for paths read recently *and* edit records for paths written recently. Collect the set of active paths.
3. For each active path, find every pre-tail `(function_call, function_call_output)` pair whose output is a live window for that path. All such pairs are preserved — multiple live windows per path (different ranges) all carry useful context.
4. Lift those pairs out of the pre-tail region and re-insert them in the tail immediately before the first item in the tail carrying a `file_tool.path` annotation. This placement is constrained by the Anthropic Messages API requirement that the first message in the `messages` array be `user` role: `_recency_tail_items()` already advances the tail boundary forward until it lands on a user-role text message, so any annotated tail item is guaranteed to come after at least that leading user message. Inserting lifted pairs at position 0 of the tail would put an assistant `tool_use` first and break user-first; inserting before the first annotated item slots them after the tail's leading user-role prefix while keeping them adjacent to the downstream activity that uses them. The pairs retain their original `call_id`s and arguments — we move existing items rather than synthesizing fresh ones, so the conversation history accurately documents what the model originally called.
5. Compact the remainder of the pre-tail region into a summary as today.

After injection, the lifted pairs appear as a sequence of earlier tool-call rounds preceding the tail's first annotated round. This preserves the read → downstream-use relationship — lifted reads come before the recent file activity that depends on them — though not strict global chronology relative to the tail's leading conversational items, since those originally happened *after* the lifted reads but appear *before* them in the new layout. Strict global ordering would require placing lifted pairs at position 0 of the tail, which the user-first constraint above rules out. The two providers handle this differently but both work:

- `to_anthropic_messages()` groups consecutive same-role items into single messages and flushes on role change, so each lifted `(function_call, function_call_output)` pair renders as its own assistant/user message pair — i.e., its own round. Anthropic accepts this because `call_id` pairs every `tool_use` to its `tool_result` regardless of message grouping.
- `to_openai_input()` is a flat iteration with no round-grouping logic — it dumps each item in order keyed by `call_id`. Lifted items just become earlier entries in the flat input list. The OpenAI Responses API has no round structure to violate.

The model's narrative is "I did these reads earlier; here's my recent write activity," which is also more honest than claiming concurrent calls in a single round when the reads originally happened over many turns.

Subsequent reconciliation and write-refresh continue to operate on the now-tail live windows the same way they always do — there is no separate code path for lifted vs. originally-tail windows.

If multiple active paths each have lifted pairs, all are inserted contiguously before the tail's first `file_tool.path`-annotated item, preserving their original chronological order from the pre-tail region.

#### Summaries do not contain live-window file contents

`ancestor_summaries` are immutable once written: no future `reconcile_tracked_files` pass reaches into them. So if any live window's current file body lands in a summary, that body is fossilized and will drift out of sync with the on-disk truth that subsequent live-window refreshes track. The model would then receive both the up-to-date live windows *and* a stale summary description of the same files, and it would have no way to know which is current.

To prevent that, both web harnesses' `_summarize_and_compact` pass the snapshot through `_strip_live_window_bodies(partition)` before generating the summarization input. The helper deep-copies the partition and rewrites every live window — *not* just those whose path is active in the recency tail — so current file contents cannot leak into the summary regardless of activity classification.

The invariant is broad on purpose: once content lands in a summary, there is no later reconciliation pass that can repair it, so any live window in the summarization input is a candidate for fossilization. Restricting the strip to "active paths" would leave inactive ones as a leak.

Two stripping shapes, by output prefix:

- Ordinary `FILE ... status=live` read results are replaced wholesale with `[Live tracked file: <path> — current contents are tracked via the live-window mechanism on subsequent turns, not summarized here.]`. The diagnostic value is just the path.
- `CONFLICT` and `RANGE_ERROR` results keep their two-line diagnostic header (which describes what the model attempted and the failure mode) and have only the embedded `Current view ...` body replaced with the same placeholder. The summarizer still sees "model tried to write with stale revision" or "model passed an out-of-range edit," but never the actual file contents.

Edit records, tombstones (`file_tool.status="stale"`), function_call items, and message items are preserved verbatim. Edit records and tombstones already document file activity without embedding current contents; the rest are not file-related. The original `snapshot` is not mutated — the lift logic in `_compact_partition` continues to use it after the summarizer call returns.

#### Prefix check tolerates live-window refresh

Compaction runs as a background task: a snapshot is captured at compaction-trigger time, then a summary is computed and an atomic Redis swap installs the new partition. Before the swap, `_compact_partition` checks that `current_partition` (freshly loaded from Redis) is still a strict prefix-extension of `snapshot`. The strict-equality check would normally falsify the prefix the moment any concurrent request between the snapshot and the swap ran `reconcile_tracked_files` or applied a `file_tool` write — both of those mutate prior live windows in-place, even when no semantic change occurred.

The prefix check therefore uses `_items_equal_mod_live_windows`, which treats the mutation-fields of live windows (`output`, `file_tool.revision`, `file_tool.view_end_line`) as comparison-irrelevant while keeping the stable identity (`type`, `call_id`, `id`, `file_tool.path`, `file_tool.status`, `file_tool.view_start_line`) and full equality on every other item kind. Substantive changes still skip the swap: a tombstone transition (`file_tool.status` live → stale), a new appended item that lengthens the prefix slice, an edit-record difference, or any non-live-window mutation all fall through to inequality. The helper lives in `web_v1` and operates on annotation-key strings only, so it does not introduce a `web_v1 → tools_v1` import.

The trade-off this accepts: lifted live windows in the swapped partition come from `snapshot.items` and may briefly carry pre-refresh `output` and `file_tool.revision` if a concurrent request had refreshed those same windows in `current_partition`. This is a one-turn lag rather than a regression — the next request's `reconcile_tracked_files` re-reads the on-disk state and refreshes those windows back to current. The same lag exists today for any compacted partition restored from Redis or ES, and is the rationale for the contract documented under "Reconciliation before each turn."

### Conflict and edge cases

**Indexing.** All line numbers are 1-based and inclusive. `start_line=1` refers to the first line of the file.

**Empty files.** Reading returns a `function_call_output` whose `output` carries the FILE header with `line_count=0` and no body lines. Writes against an empty file are valid: `insert_lines` with `start_line=1` is the only well-defined write; `replace_lines` and `delete_lines` against an empty file return an out-of-range error (see below).

**EOF insertion.** `insert_lines` with `start_line = line_count + 1` appends to the end of the file. Any larger value returns an out-of-range error.

**Out-of-range edits.** When `start_line` or `end_line` falls outside `[1, line_count]` (with the EOF-insertion exception above), the call returns a result of the same shape as a conflict, but with a header indicating the range error. No filesystem write occurs.

**Conflict result format.** Returned when `expected_revision` does not match the current on-disk revision. The item is a `function_call_output` whose `output` is:

```text
CONFLICT path=<absolute path> expected_revision=<x> current_revision=<y>
The file changed since the revision you read. Re-read before retrying.
Current view (lines <s>-<e> of <line_count>):
<line-numbered view from start_line up to max_lines>
```

The item carries the same `prokaryotes_annotations` as a successful read for the refreshed view (`file_tool.path`, `file_tool.revision=<current>`, `file_tool.status="live"`, `file_tool.view_start_line`, `file_tool.view_end_line`), so it doubles as a fresh tracked snapshot. The model can immediately retry the write using `current_revision`.

## Pseudocode

### `FileTool.__init__` and `call(...)`

```python
class FileTool(FunctionToolCallback):
    # Class-level: shared across all FileTool instances in this process.
    _path_locks: dict[str, asyncio.Lock] = {}

    def __init__(self, context_partition: ContextPartition, workspace_root: Path = Path("/")):
        self._partition = context_partition
        self._workspace_root = workspace_root
        self._lock = asyncio.Lock()  # serializes call() within this request

    @classmethod
    def _get_path_lock(cls, path: str) -> asyncio.Lock:
        lock = cls._path_locks.get(path)
        if lock is None:
            lock = cls._path_locks.setdefault(path, asyncio.Lock())
        return lock

    async def call(self, arguments: str, call_id: str) -> ContextPartitionItem:
        async with self._lock:
            payload = json.loads(arguments)
            action = payload["action"]
            path = self._resolve_path(payload["path"])
            if action == "read":
                return await self._do_read(call_id, path, payload)
            if action in {"replace_lines", "insert_lines", "delete_lines"}:
                return await self._do_write(call_id, path, action, payload)
            raise ValueError(f"Unsupported action: {action}")

    async def _do_write(self, call_id, path, action, payload) -> ContextPartitionItem:
        expected_revision = payload.get("expected_revision")
        if not expected_revision:
            return _error_item(call_id, "expected_revision is required ...")
        path_lock = self._get_path_lock(str(path))
        async with path_lock:
            result_item, current_text, current_revision = await asyncio.to_thread(
                self._locked_write_transaction, call_id, path, action, payload, expected_revision,
            )
        # Always refresh — including on conflict and range-error results — so older live
        # windows for this path move to the same revision the just-built result item
        # reports. _refresh_live_windows is a no-op for items already at this revision.
        _refresh_live_windows(self._partition.items, str(path), current_text, current_revision)
        return result_item

    def _locked_write_transaction(self, call_id, path, action, payload, expected_revision):
        # Runs inside asyncio.to_thread. Returns (item, current_text, current_revision)
        # for all three result kinds; current_* describes the file's post-transaction
        # on-disk state.
        with open(path, "r+", encoding="utf-8") as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
            original_text = fp.read()
            current_revision = sha256(original_text.encode("utf-8")).hexdigest()
            line_count = _count_lines(original_text)
            if expected_revision != current_revision:
                return self._build_view_carrying_item(... CONFLICT ...), original_text, current_revision
            if not _range_is_valid(action, payload, line_count):
                return self._build_view_carrying_item(... RANGE_ERROR ...), original_text, current_revision
            updated_text = _apply_line_edit(original_text, action, payload)
            fp.seek(0); fp.truncate(); fp.write(updated_text); fp.flush()
            new_revision = sha256(updated_text.encode("utf-8")).hexdigest()
        # Edit record carries only `file_tool.path` — frozen audit trail, never refreshed.
        return ContextPartitionItem(
            call_id=call_id,
            output=render_edit_record(...),
            prokaryotes_annotations={"file_tool.path": str(path)},
            type="function_call_output",
        ), updated_text, new_revision
```

The advisory lock is released when the `with open(...)` block exits and closes the descriptor; an explicit `LOCK_UN` is unnecessary. The per-path `asyncio.Lock` is released after the threaded transaction returns; refresh runs outside it because it mutates only this request's in-memory partition.

### Harness reconciliation helper

Updates externally changed file items in-place. Each live item has a matching `function_call` already in the partition, so mutating `item.output` and annotations is valid on both the Anthropic and OpenAI paths.

```python
# Module-level functions in prokaryotes/tools_v1/file_tool.py.

async def reconcile_tracked_files(context_partition: ContextPartition) -> None:
    # Group all live items by path; multiple live items per path can co-exist.
    paths_with_live_items: set[str] = set()
    for item in context_partition.items:
        ann = item.prokaryotes_annotations or {}
        if item.type == "function_call_output" and ann.get("file_tool.status") == "live":
            paths_with_live_items.add(ann["file_tool.path"])

    for path in paths_with_live_items:
        try:
            current_text = await asyncio.to_thread(Path(path).read_text, encoding="utf-8")
        except (FileNotFoundError, IsADirectoryError, PermissionError, UnicodeDecodeError) as exc:
            for item in context_partition.items:
                ann = item.prokaryotes_annotations or {}
                if ann.get("file_tool.path") == path and ann.get("file_tool.status") == "live":
                    item.prokaryotes_annotations["file_tool.status"] = "stale"
                    item.output = render_tombstone_marker(path, type(exc).__name__)
            continue

        current_revision = sha256(current_text.encode("utf-8")).hexdigest()
        _refresh_live_windows(context_partition.items, path, current_text, current_revision)


def _refresh_live_windows(items, path: str, text: str, revision: str) -> None:
    """Re-render every live window for `path` against `text`/`revision`. Shared by FileTool writes
    and reconcile_tracked_files. Items already at `revision` are left alone."""
    for item in items:
        ann = item.prokaryotes_annotations or {}
        if ann.get("file_tool.path") != path or ann.get("file_tool.status") != "live":
            continue
        if ann["file_tool.revision"] == revision:
            continue
        start_line = int(ann["file_tool.view_start_line"])
        end_line, rendered = render_view(text, start_line, FileTool.max_lines)
        item.output = rendered
        ann["file_tool.revision"] = revision
        ann["file_tool.view_end_line"] = str(end_line)
```

In `post_chat()` (both harnesses), after `sync_context_partition()`:

```python
from prokaryotes.tools_v1.file_tool import reconcile_tracked_files

context_partition = await self.sync_context_partition(conversation)
await reconcile_tracked_files(context_partition)

# ... assemble developer_message_parts ...
```

## Files Likely To Change

- `prokaryotes/tools_v1/file_tool.py` — new tool implementation; module-level `reconcile_tracked_files()` and `_refresh_live_windows()` helpers (called from each harness's `post_chat()`)
- `prokaryotes/tools_v1/README.md` — document the new tool
- `prokaryotes/api_v1/models.py` — add `prokaryotes_annotations: dict[str, str] | None` to `ContextPartitionItem`; remove the `text_preamble` field; remove the `text_preamble` branch in `to_anthropic_messages()` and the synthetic-assistant-message branch in `to_openai_input()`; update `to_openai_input()`'s `model_dump` exclude to `{"prokaryotes_annotations"}`
- `prokaryotes/openai_v1/__init__.py` — remove the `tool_preamble` accumulator and `text_preamble` assignment in `handle_response_stream_event()`
- `prokaryotes/anthropic_v1/__init__.py` — remove the `text_preamble=...` argument in `stream_turn()`'s `function_call` item construction
- `prokaryotes/openai_v1/web_harness.py` — register `FileTool`, call `reconcile_tracked_files()`
- `prokaryotes/anthropic_v1/web_harness.py` — same as OpenAI harness
- `prokaryotes/web_v1/__init__.py` — modify `_compact_partition()` to lift pre-tail live windows for tail-active paths into the new tail (`_lift_active_live_windows`), and to use `_items_equal_mod_live_windows` for its prefix-equality check so concurrent live-window refresh between snapshot and swap does not falsify the prefix. Add `_strip_live_window_bodies` (used by both harnesses' `_summarize_and_compact`) so live-window file contents cannot land in `ancestor_summaries`. All three helpers inspect `prokaryotes_annotations` keys as plain strings, so no `tools_v1` import is needed here.
- `prokaryotes/anthropic_v1/web_harness.py` and `prokaryotes/openai_v1/web_harness.py` — `_summarize_and_compact()` runs `_strip_live_window_bodies(snapshot)` before generating the summarization input, so the original snapshot remains untouched for the subsequent lift step in `_compact_partition`.
- `scripts/static/ui.js` — add `formatFileToolCallMarkdown` and register it in `formatToolCallMarkdown`
- overlay tests under `project/wip/file-tool-tracked-context/overlay/tests/` once implementation begins

## Infrastructure Changes

- No new external service is expected.
- No changes to Redis or Elasticsearch persistence: `ContextPartition` gains no new fields, and the new `ContextPartitionItem` metadata fields are stored inside the existing `items_json` blob.
- `FileTool` accepts a `workspace_root` parameter (default `/`) so path sandboxing can be tightened in future deployments without code changes. The app currently runs in a disposable Docker container, so the default of `/` is appropriate.

## Open Questions

None currently.

## Recommended Initial Direction

Implement this as a single-commit feature with:

- a new `FileTool` taking `context_partition` at construction time, using action-based JSON arguments,
- line-numbered reads and line-range write operations with required `expected_revision` on all writes,
- file tracking derived from live `function_call_output` items in the existing partition — no new registry field on `ContextPartition`,
- module-level `reconcile_tracked_files()` in `tools_v1/file_tool.py`, called from both harnesses' `post_chat()` after `sync_context_partition()` (kept out of `WebBase` to avoid a `web_v1 → tools_v1` dependency),
- three distinct output kinds, each with one job: **live windows** (read / conflict / refreshed earlier window — refreshed in-place on writes and external changes), **edit records** (write result — frozen diff carrying only `file_tool.path`, never refreshed, serves as the model's audit trail and as the path-activity signal for compaction), and **tombstones** (file became inaccessible — marked stale on prior live windows in-place),
- compaction-aware preservation: at compaction time, all pre-tail live windows for paths active in the recency tail (any item with a `file_tool.path` annotation) are lifted into the tail as earlier rounds preceding the tail's events, preserving original `call_id`s and arguments. The compaction prefix check uses `_items_equal_mod_live_windows` so a concurrent request that refreshed prior live windows in-place doesn't falsify the prefix; lifted windows that briefly trail current FS state are repaired by the next request's `reconcile_tracked_files`. Before summarization, `_strip_live_window_bodies` rewrites every live-window output in the snapshot copy so current file contents cannot fossilize into `ancestor_summaries` (CONFLICT / RANGE_ERROR diagnostic headers are kept; embedded `Current view` bodies and ordinary `FILE` views are replaced with a path-only placeholder),
- two-layer cross-request write safety: per-`FileTool` `asyncio.Lock` serializes `call()` within one request; a class-level per-path `asyncio.Lock` plus an OS-level `fcntl.flock(LOCK_EX)` on the target file's own descriptor inside the threaded write transaction makes the read → revision-check → write atomic across cooperating writers on the same host; external (non-`file_tool`) writers are not coordinated and their changes are detected on the next read or reconciliation; all filesystem I/O routed through `asyncio.to_thread`,
- explicit conflict and out-of-range result formats that include the current revision and a refreshed view so the model can retry deterministically,
- a single `prokaryotes_annotations: dict[str, str] | None` field on `ContextPartitionItem` (excluded from `to_openai_input()`) used by `FileTool` with dot-namespaced `file_tool.*` keys,
- removal of the existing `text_preamble` field on `ContextPartitionItem` (and its replay logic in `to_anthropic_messages()` / `to_openai_input()` plus the `tool_preamble` accumulator in the OpenAI client) — the `progress_message` stream already covers the streaming UX and removing `text_preamble` eliminates a corner case in the compaction lift where lifted `function_call` items would otherwise carry orphaned narration into a new round.

That delivers the main benefit of a canonical synchronized file representation without requiring client-side protocol changes or new persistence infrastructure.
