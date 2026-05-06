# Review: file-tool-tracked-context

Reviewed: design doc `project/wip/file-tool-tracked-context/README.md` against the current repo (`prokaryotes/api_v1/models.py`, both `_v1` clients and harnesses, `prokaryotes/web_v1/__init__.py`, `prokaryotes/tools_v1/`, `scripts/static/ui.js`) and the overlay (`overlay/prokaryotes/...` and `overlay/tests/`).

## Bugs

### B1. Compaction summaries can fossilize stale file state

The design's core promise is that prior file reads are **live windows** that keep tracking current file content across turns and even across external edits. That is implemented through in-place refresh during reconciliation and writes.

But compaction still asks the summarizer to "Preserve key decisions, facts, code produced, and tool call outcomes" using the full snapshot (`prokaryotes/openai_v1/web_harness.py`, `_summarize_and_compact()` at lines 53-58; `prokaryotes/anthropic_v1/web_harness.py`, `_summarize_and_compact()` at lines 46-49). If a live window's file content gets summarized into `ancestor_summaries`, that summary becomes immutable background memory. Later live-window refreshes will update the raw tool outputs, but not the compacted summary, so the model can end up with stale file state reintroduced through the summary path.

The doc should define an explicit invariant here: summaries must not restate current file contents for tracked active paths, or compaction must strip / rewrite live-window file bodies before summarization. Otherwise the "canonical synchronized file representation" is only true until the next compaction.

## Non-blocking

### N1. Newline behavior is not specified in the README

The design is precise about line indexing and revision hashes, but it does not say what happens to trailing newlines on write. The overlay implementation makes a concrete choice in `_apply_line_edit` (`overlay/prokaryotes/tools_v1/file_tool.py:428-443`): preserve a trailing newline when the original file had one, and give newly non-empty files a trailing newline when editing from empty.

That choice should be promoted into the README. Without it, a future implementation could make a different normalization decision and still appear to satisfy the line-edit protocol while producing different revisions and on-disk bytes.

### N2. Conflict / range-error headers need an explicit lifetime rule

The README says conflict and range-error results carry live-window annotations so they can double as fresh tracked snapshots, which is good. What it does not say explicitly is what happens on the *next* refresh.

The overlay's `_refresh_live_windows()` (`overlay/prokaryotes/tools_v1/file_tool.py:475-504`) re-renders those items back into ordinary `FILE ... status=live` windows once the file changes again. So `CONFLICT ...` and `RANGE_ERROR ...` are transient headers, not stable historical records. That behavior is coherent, but surprising enough that it belongs in the spec; otherwise a reader may reasonably assume those diagnostic outputs persist in history.

### N3. "No other client protocol changes are required" is only true for model context

That statement is true for server-side model reconstruction, but not for human-visible history. The current UI stores streamed `progress_message` / `tool_call` activity locally per assistant node and never rehydrates historical tool outputs from persisted server state (`scripts/static/ui.js:958-1052`). So the harness can mutate earlier `function_call_output` items between turns and the model will see the refreshed version, while the user-facing chat transcript will not.

If the intention is strictly model-side canonicalization, the doc should say so. If the intention is that users can inspect the same evolving live windows the model sees, this feature needs a follow-up retrieval / rendering path beyond the current stream-only activity UI.

## Looks correct

- The proposal is grounded in the real architecture: `FunctionToolCallback`, `ContextPartitionItem`, provider-specific serialization, pre-turn reconciliation, and compaction hooks all match the current code layout.
- Removing `text_preamble` is well motivated. In the current code it exists only to reconstruct pre-tool narration (`prokaryotes/api_v1/models.py`, `to_anthropic_messages()` at lines 161-179 and `to_openai_input()` at lines 203-220; provider collection in `prokaryotes/openai_v1/__init__.py` and `prokaryotes/anthropic_v1/__init__.py`). The README correctly identifies that lifted tool-call pairs would otherwise drag old narration into a new position during compaction.
- The overlay test surface is meaningful, not decorative. The documented overlay Python tests pass when run with the overlay `PYTHONPATH`, and the overlay JS formatter tests pass as-is.

## Suggested next step

Tighten the compaction section first so summaries cannot reintroduce stale file contents for tracked files. After that, add one short spec paragraph for newline policy and one for the transient nature of conflict / range-error headers. The UI visibility note can likely stay as a follow-up unless the product intent is specifically to expose live windows to users.
