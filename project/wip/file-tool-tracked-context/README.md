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

Both LLM clients dispatch tool callbacks within a round concurrently (`asyncio.create_task` + `asyncio.gather`). `FileTool` therefore holds an `asyncio.Lock` that serializes its own `call()` invocations within a request — the lock guards both the in-memory partition mutation and the surrounding I/O so concurrent file ops don't interleave their refresh-in-place loops. Filesystem-level write-write conflicts on the same path are still possible across requests but are caught by the `expected_revision` check and surfaced as a conflict result. All filesystem I/O uses `asyncio.to_thread` to avoid blocking the event loop.

The tool exposes a single `action` enum parameter. Proposed values:

- `read` — return a line-numbered view from `start_line` up to `max_lines` lines.
- `replace_lines` — replace an inclusive line range with new text.
- `insert_lines` — insert text at a given line boundary.
- `delete_lines` — delete an inclusive line range.

### 2. File tracking via existing partition items

Rather than maintaining a separate tracked-file registry, the harness derives which files are currently tracked by scanning `context_partition.items` for `function_call_output` items whose `prokaryotes_annotations` carry `"file_tool.status": "live"`. Each live item IS a tracking record — it carries the current `file_tool.revision`, `file_tool.view_start_line`, and `file_tool.view_end_line` annotations directly.

Multiple live items per path can co-exist: e.g., the model reads lines 1–200 and then lines 201–400 of the same file. Both items remain live and represent independently useful views into the current file.

A live item is a **live window**, not a static snapshot. Whenever the file changes — whether through a subsequent `file_tool` write or an external edit detected at reconciliation time — every live window for that path is refreshed in-place: its stored view range is re-rendered against the new content, and its revision and end-line annotation are updated. The model therefore sees earlier `function_call_output` items in its conversation history change between turns to track the current file state. This is the central departure from the default assumption that tool outputs are immutable, and the system message guidance below makes it explicit to the model.

Compaction does require explicit handling so live windows for actively-worked files survive: the recency tail's `file_tool.path` annotations identify "active paths", and pre-tail live windows for those paths are lifted into the tail at compaction time. See "Compaction behavior" for the full mechanism.

### 3. Canonical file output protocol

`FileTool` produces three distinct kinds of `function_call_output` text. Each kind has one job; the harness identifies them by their leading marker and (where applicable) by `prokaryotes_annotations`.

**Live window** (read result, conflict result, refreshed earlier window). Carries `prokaryotes_annotations` with `file_tool.status="live"` so it is picked up by the live-window scan and refreshed by future changes.

```text
FILE path=project/wip/README.md revision=abc123 status=live lines=1-40 line_count=87
1 | # Feature Planning and Implementation Process
2 | ...
```

**Edit record** (write result). A frozen historical record of what changed. Carries `prokaryotes_annotations = {"file_tool.path": str(path)}` — just the path, with no `file_tool.status`, so it is *not* a live window, is never refreshed, and survives across turns so the model can retrace its edits or revert them. The path annotation is what compaction uses to identify "active paths" worth preserving live windows for (see "Compaction behavior" below). Removed-line numbers reference the file before the edit; added-line numbers reference the file after.

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

**Important**: `to_openai_input()` uses `item.model_dump(exclude_none=True, exclude={"text_preamble"})`, which would include `prokaryotes_annotations` in the dict sent to the OpenAI Responses API. It must be added to the `exclude` set: `exclude={"text_preamble", "prokaryotes_annotations"}`. `to_anthropic_messages()` is not affected because it constructs its output from explicit field accesses rather than `model_dump()`.

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

1. Resolve `path` against `workspace_root`, verify it does not escape it, and load the current file contents using UTF-8 encoding via `asyncio.to_thread`.
2. Compute current `revision` as `sha256(text.encode("utf-8")).hexdigest()`.
3. If `expected_revision` does not match current revision, return a conflict result instead of writing (see "Conflict and edge cases" below).
4. Apply line-based mutation.
5. Write updated contents back to disk via `asyncio.to_thread`.
6. Recompute revision and line count.
7. Refresh all live items for this path in-place using the same logic as reconciliation: re-render each window from its stored `file_tool.view_start_line` against the new content, replace `output` with the fresh rendered view, and update `file_tool.revision` and `file_tool.view_end_line` in the annotations. The items remain `"live"`.
8. Return a frozen edit record (see "Canonical file output protocol") showing the lines removed (in old line numbers) and added (in new line numbers), plus the revision and `line_count` transition. The edit record carries `prokaryotes_annotations = {"file_tool.path": str(path)}` — just the path, no `file_tool.status` — so it is not a live window and is never refreshed; it serves as the model's audit trail so it can retrace or undo what it just did even after live windows have moved on, and the path annotation lets compaction recognize this path as active.

### Reconciliation before each turn

Add `reconcile_tracked_files()` to `WebBase`. Call it from `post_chat()` in each harness after `sync_context_partition()` returns and before the developer message is assembled.

Each live file item already has a matching `function_call` item in the partition, so updating `item.output` and annotations in-place is valid — no orphaned items are created.

1. Scan `context_partition.items` for `function_call_output` items whose `prokaryotes_annotations` carry `"file_tool.status": "live"`, grouping all live items by `file_tool.path`.
2. For each path, attempt to read the file once from disk with UTF-8 encoding and compute current revision as `sha256(text.encode("utf-8")).hexdigest()`. Use `asyncio.to_thread` for the read.
3. If the read raises `FileNotFoundError`, `IsADirectoryError`, `PermissionError`, or `UnicodeDecodeError`, mark every live item for that path stale in-place: set `file_tool.status = "stale"` and replace `item.output` with a tombstone marker that names the failure mode (e.g. `"FILE path=<p> status=stale [no longer accessible: <reason>]"`). Continue to the next path.
4. For each live item belonging to a successfully-read path:
   - if the current revision matches `item.prokaryotes_annotations["file_tool.revision"]`, leave the item alone;
   - otherwise update the item in-place: re-render the file view from `file_tool.view_start_line` up to `FileTool.max_lines` lines, replace `item.output` with the fresh rendered view, and update `file_tool.revision` and `file_tool.view_end_line` in the annotations. The item remains `"live"`.

### Compaction behavior

Live windows must survive compaction for files the model is still actively working on, otherwise the live-window framing breaks down: a file read at turn 1 would silently lose its tracking on the next compaction even if the model is mid-edit at turn 30. The compaction logic uses the recency tail as a signal of which paths are active and lifts pre-tail live windows for those paths into the tail.

1. Compute the recency tail as today.
2. Scan the recency tail for any item carrying a `file_tool.path` annotation. This includes both live windows for paths read recently *and* edit records for paths written recently. Collect the set of active paths.
3. For each active path, find every pre-tail `(function_call, function_call_output)` pair whose output is a live window for that path. All such pairs are preserved — multiple live windows per path (different ranges) all carry useful context.
4. Lift those pairs out of the pre-tail region and re-insert them in the tail immediately before the first item in the tail carrying a `file_tool.path` annotation. The pairs retain their original `call_id`s and arguments — we move existing items rather than synthesizing fresh ones, so the conversation history accurately documents what the model originally called.
5. Compact the remainder of the pre-tail region into a summary as today.

After injection, the message flow looks like the model emitted concurrent tool calls in that round: the original write (or read) plus the lifted reads, all sharing the same assistant turn for `function_call`s and the same user turn for `function_call_output`s. Both providers accept this — `to_anthropic_messages` and `to_openai_input` already handle multiple tool calls per round, and `call_id` pairing keeps everything matched. The model sees a coherent narrative ("this round, I read X and Y; then I wrote to X"), even though the reads originally happened earlier.

Subsequent reconciliation and write-refresh continue to operate on the now-tail live windows the same way they always do — there is no separate code path for lifted vs. originally-tail windows.

If multiple active paths share an edit record's tool round, all carried-read pairs for all paths are inserted as a single batch before the earliest `file_tool.path`-annotated item in the tail, so they all appear in that single round.

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
def __init__(self, context_partition: ContextPartition, workspace_root: Path = Path("/")):
    self._partition = context_partition
    self._workspace_root = workspace_root
    self._lock = asyncio.Lock()  # serializes call() within this request

async def call(self, arguments: str, call_id: str) -> ContextPartitionItem:
    async with self._lock:
        payload = json.loads(arguments)
        action = payload["action"]
        path = self._resolve_path(payload["path"])  # resolves within workspace_root

        if action == "read":
            text = await asyncio.to_thread(path.read_text, encoding="utf-8")
            revision = sha256(text.encode("utf-8")).hexdigest()
            start_line = payload["start_line"] or 1
            end_line, rendered = render_view(text, start_line, self.max_lines)
            return self._build_live_item(call_id, path, revision, "live", start_line, end_line, rendered)

        if action in {"replace_lines", "insert_lines", "delete_lines"}:
            original_text = await asyncio.to_thread(path.read_text, encoding="utf-8")
            current_revision = sha256(original_text.encode("utf-8")).hexdigest()
            line_count = original_text.count("\n") + (0 if original_text.endswith("\n") or not original_text else 1)

            if payload["expected_revision"] != current_revision:
                return self._build_conflict_item(call_id, path, payload["expected_revision"], current_revision, original_text)
            if not _range_is_valid(action, payload, line_count):
                return self._build_range_error_item(call_id, path, current_revision, original_text, payload, line_count)

            updated_text = apply_line_edit(original_text, payload)
            await asyncio.to_thread(path.write_text, updated_text, encoding="utf-8")
            new_revision = sha256(updated_text.encode("utf-8")).hexdigest()

            # Refresh all prior live windows for this path in-place.
            # `_refresh_live_windows` is the same helper used by reconcile_tracked_files().
            _refresh_live_windows(self._partition.items, str(path), updated_text, new_revision)

            # Edit record is a frozen historical output — never refreshed.
            # Carries only `file_tool.path` so compaction can detect path activity in the tail.
            edit_record = render_edit_record(
                action=action,
                path=str(path),
                old_revision=current_revision,
                new_revision=new_revision,
                old_text=original_text,
                new_text=updated_text,
                payload=payload,
            )
            return ContextPartitionItem(
                call_id=call_id,
                output=edit_record,
                prokaryotes_annotations={"file_tool.path": str(path)},
                type="function_call_output",
            )

        raise ValueError(f"Unsupported action: {action}")
```

### Harness reconciliation helper

Updates externally changed file items in-place. Each live item has a matching `function_call` already in the partition, so mutating `item.output` and annotations is valid on both the Anthropic and OpenAI paths.

```python
async def reconcile_tracked_files(self, context_partition: ContextPartition) -> None:
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
context_partition = await self.sync_context_partition(conversation)
await self.reconcile_tracked_files(context_partition)

# ... assemble developer_message_parts ...
```

## Files Likely To Change

- `prokaryotes/tools_v1/file_tool.py` — new tool implementation
- `prokaryotes/tools_v1/README.md` — document the new tool
- `prokaryotes/api_v1/models.py` — new `prokaryotes_annotations: dict[str, str] | None` field on `ContextPartitionItem`, and updated `to_openai_input()` to exclude it alongside `text_preamble`
- `prokaryotes/openai_v1/web_harness.py` — register `FileTool`, call `reconcile_tracked_files()`
- `prokaryotes/anthropic_v1/web_harness.py` — same as OpenAI harness
- `prokaryotes/web_v1/__init__.py` — `reconcile_tracked_files()` helper on `WebBase`; modify `_compact_partition()` to lift pre-tail live windows for tail-active paths into the new tail
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
- `reconcile_tracked_files()` on `WebBase`, called from both harnesses after `sync_context_partition()`,
- three distinct output kinds, each with one job: **live windows** (read / conflict / refreshed earlier window — refreshed in-place on writes and external changes), **edit records** (write result — frozen diff carrying only `file_tool.path`, never refreshed, serves as the model's audit trail and as the path-activity signal for compaction), and **tombstones** (file became inaccessible — marked stale on prior live windows in-place),
- compaction-aware preservation: at compaction time, all pre-tail live windows for paths active in the recency tail (any item with a `file_tool.path` annotation) are lifted into the tail as additional concurrent tool calls in the existing round, preserving original `call_id`s and arguments,
- per-`FileTool` `asyncio.Lock` to serialize `call()` within a request; all filesystem I/O routed through `asyncio.to_thread`,
- explicit conflict and out-of-range result formats that include the current revision and a refreshed view so the model can retry deterministically,
- a single `prokaryotes_annotations: dict[str, str] | None` field on `ContextPartitionItem` (excluded from `to_openai_input()`) used by `FileTool` with dot-namespaced `file_tool.*` keys.

That delivers the main benefit of a canonical synchronized file representation without requiring client-side protocol changes or new persistence infrastructure.
