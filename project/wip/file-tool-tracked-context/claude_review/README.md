# Review: file-tool-tracked-context

Reviewed: design doc `project/wip/file-tool-tracked-context/README.md` against the current repo (`prokaryotes/api_v1/models.py`, both `_v1` clients and harnesses, `prokaryotes/web_v1/__init__.py`, `prokaryotes/tools_v1/`, `prokaryotes/search_v1/context_partitions.py`) and the overlay (`overlay/prokaryotes/...` and `overlay/tests/`).

## Bugs

### B1. Cross-request write race is overstated as "caught"

Doc §"Core Concepts and New Abstractions" → "1. `FileTool`" paragraph 2 claims:

> Filesystem-level write-write conflicts on the same path are still possible across requests but are caught by the `expected_revision` check and surfaced as a conflict result.

This is wrong. The per-instance `asyncio.Lock` only serializes calls within one request's `FileTool` instance. Two concurrent requests instantiate independent `FileTool`s with independent locks, so the read-check-write sequence in `_do_write` (`overlay/prokaryotes/tools_v1/file_tool.py:91-134`) is not atomic relative to another request:

- R1 reads → rev A; R2 reads → rev A.
- Both pass `expected_revision == "A"`.
- R1 writes → on-disk rev B; R2 writes → on-disk rev C, silently overwriting B.

Both `EDITED` records claim success, but only one set of changes survives. Either the doc must acknowledge the race honestly, or wrap the read-check-write in an OS-level lock (`fcntl.flock` on the path) or atomic rename. As written, the doc misleads a reader about the safety guarantee.

### B2. Lift insertion point is missing the "back up to paired call" rule

Doc §"Compaction behavior" step 4:

> Lift those pairs out of the pre-tail region and re-insert them in the tail immediately before the first item in the tail carrying a `file_tool.path` annotation.

Only `function_call_output` items carry annotations — `function_call` items never do. So the *first annotated tail item* is always a `function_call_output`, and inserting before it splits an adjacent `(function_call, function_call_output)` pair, leaving the original pair's call dangling above the lifted block while the original output sits below it. That breaks `to_anthropic_messages` role-grouping and is unconventional for OpenAI too.

The overlay (`overlay/prokaryotes/web_v1/__init__.py:86-108`, `tail_function_call_idx_by_call_id`) correctly backs up to the paired `function_call`. The design doc must describe this step — without it, a future re-implementer reading only the doc would write the broken version.

### B3. `prokaryotes_annotations` exclusion rationale is incomplete

Doc §"Data Model Changes" → "`ContextPartitionItem`" — paragraph after the table — explains why `to_openai_input()` must exclude `prokaryotes_annotations` (the dict would leak into the API payload). True, but the doc should also state the broader invariant that's now load-bearing:

> Every field on `ContextPartitionItem` is either a known provider field, or excluded from `to_openai_input` and not read by `to_anthropic_messages`.

This matters because file_tool now mutates items in-place between turns; any future addition of an internal field on `ContextPartitionItem` (e.g. a per-item flag) needs the same exclusion treatment, and the reviewer of that change needs to know to look. Today the rationale reads like a one-off note rather than a rule.

### B4. Compaction prefix check is meaningfully more fragile

`web_v1/__init__.py:282-291` (and the overlay copy at the same lines) uses Pydantic equality on full items:

```python
current_partition.items[:len(snapshot.items)] != snapshot.items
```

Today items are append-only and equality is stable. With file_tool, *any* concurrent request that runs `reconcile_tracked_files` or any file_tool write between the snapshot and the swap will mutate `output` and `prokaryotes_annotations` of pre-existing items in-place, falsifying the prefix equality. The swap then silently skips and the compaction work is wasted.

This is acceptable as "best-effort compaction" but isn't called out in the doc. At minimum the doc should state: *the swap will skip more often when file_tool is active; the summary work is wasted but no state is corrupted.* If the design wants the swap to succeed despite in-place mutations, the prefix comparison must be relaxed (e.g., compare on `(call_id, type)` only, or on a hash of the message-only projection).

## Non-blocking

### N1. `max_lines = 200` lives only in code

Doc §"Tool schema" mentions a `max_lines` class constant on `FileTool` but never names a default. Overlay picks 200 (`overlay/prokaryotes/tools_v1/file_tool.py:21`). Pin the value in the doc so it's part of what the reviewer is approving.

### N2. `render_view` signature drift

Doc §"Pseudocode" → "FileTool.__init__ and call(...)" shows `end_line, rendered = render_view(text, start_line, self.max_lines)`. Overlay returns `(end_line, line_count, view_lines)`. The pseudocode should be updated; the overlay is the source of truth.

### N3. Empty / past-EOF view encoding

Doc §"Canonical file output protocol" gives the live-window header for the normal case (`lines=1-40 line_count=87`) but doesn't define what the header looks like when `start_line` is past EOF. Overlay returns `end_line = start_line - 1` and `render_live_window` emits a header without the body — so the model would see something like `lines=200-199 line_count=87`. Add one sentence so this isn't surprising.

### N4. `path.write_text` is not crash-atomic

Truncate-then-write — a crash mid-write leaves a partial file, which the next reconciliation will hash and treat as the new canonical content. Probably acceptable in a disposable container, but the doc claims revision-based safety throughout; a brief "writes are not crash-atomic" note is honest and cheap.

### N5. `shell_command` still writes files

The system-message guidance "use `file_tool` instead of `shell_command` for routine file reads/edits" is advisory only. A model that uses `sed -i` won't trip reconciliation until the *next* turn, and a model that mixes the two within one turn won't get its earlier `read` outputs refreshed mid-round. The doc should call out the model-behavior risk explicitly and consider whether evals should cover it.

### N6. Live-window framing depends on model cooperation

The system message tells the model to treat earlier `function_call_output` items as semantically *current*, even though training treats those outputs as historical events with stable line numbers. This is the most novel piece of the design; it deserves a "here's how we'll know if it's working" subsection — for example, an eval task that issues N reads then a write and checks that the model's next edit uses post-write line numbers, plus a metric on `CONFLICT` rate. As written it reads like an assumption.

### N7. `text_preamble` removal misses a doc site

Overlay removes `text_preamble` cleanly from the model and both clients. But `prokaryotes/CLAUDE.md` (the codebase overview, not in the overlay) still documents `text_preamble`'s role under "Streaming with tool-call continuation." The doc says *"a single-commit feature"* — the CLAUDE.md update belongs in §"Files Likely To Change" too, otherwise the codebase overview will document a field that no longer exists.

### N8. `tools_v1/AGENTS.md` symlink already exists

Project convention requires `README.md`, `CLAUDE.md`, and `AGENTS.md` symlinks in any directory that has a README. The overlay updates `tools_v1/README.md`; the existing `AGENTS.md` and `CLAUDE.md` symlinks in the real `prokaryotes/tools_v1/` already point to `README.md`, so no action needed — but list this in §"Files Likely To Change" so the implementer verifies it after copying.

## Looks correct

Verified explicitly, not just unmentioned:

- The `(function_call, function_call_output)` lift preserves original `call_id`s and arguments, doesn't synthesize new items, and keeps lifted pairs adjacent (`overlay/prokaryotes/web_v1/__init__.py:_lift_active_live_windows`).
- The "edit records carry only `file_tool.path`, no `file_tool.status`" split is faithfully implemented — they are treated as path-activity signals for compaction but never refreshed (`_do_write` returns at `file_tool.py:151-156`; `_refresh_live_windows` filters on `status == "live"` at `file_tool.py:485`).
- `_resolve_path` resolves symlinks before checking workspace containment (`file_tool.py:534`), so symlink-escape is rejected.
- `to_anthropic_messages` builds blocks from explicit field accesses and never reads `prokaryotes_annotations`, so annotations don't leak to Anthropic.
- `to_openai_input` correctly excludes `prokaryotes_annotations` via `model_dump(exclude=...)` (`overlay/prokaryotes/api_v1/models.py:207`).
- ES persistence works without schema change — `model_dump_json(include={"items"})` round-trips the new annotations field for free (`prokaryotes/search_v1/context_partitions.py:135`).
- Reconciliation groups by path before reading from disk — one read per path, not per item (`reconcile_tracked_files` at `file_tool.py:316-340`).
- `raw_message_start_index` is unaffected by lift, because lifted items are `function_call`/`function_call_output` and the index counts only message items.
- Per-request `asyncio.Lock` correctly serializes one request's parallel tool dispatches into a sequence (so a model that emits two `replace_lines` in one round gets one success and one CONFLICT, not silent corruption).

## Suggested next step

B1 is the only finding with a concrete data-loss scenario. Address it (either by changing the safety claim or by adding an OS-level lock around the read-check-write) and B2 (document the back-up-to-paired-call rule), then run another review pass. The rest are doc precision and minor robustness and can ride along.
