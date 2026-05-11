# Issue: `read_lines` Does Not Detect or Discourage Spans Already Covered by an Existing Live Window

## Location

- `prokaryotes/tools_v1/file_tool.py:86-179` — `_do_read_lines` always performs disk I/O and appends a fresh live window to the partition regardless of whether the requested span is already covered
- `prokaryotes/tools_v1/file_tool.py:289-295` — `_refreshable_items()` already enumerates every live window the harness can see for the current partition, including pending same-round items not yet appended
- `prokaryotes/tools_v1/file_tool.py:551-557` — system-message guidance the model is expected to follow
- `prokaryotes/tools_v1/file_tool.py:1095-1102` — `_has_transient_file_diagnostic` currently conflates "re-render on refresh" with "don't trust as stable coverage"

The relevant `system_message_parts` lines say:

> - You SHOULD plan `read_lines` calls so each file is covered by clean, contiguous **live windows**.
>   - Avoid fragmented or overlapping ranges.
>   - Do not call `read_lines` again for a span already covered by an existing live window in your current context.
>   - If compaction removed that earlier live window from context, you may read that span again.

## Problem

**Severity: Medium** — context bloat and missing corrective signal, not semantically wrong output.

The system message tells the model to avoid re-reading a span that is already covered by an existing live window, but the tool does not detect or signal redundancy. Every `read_lines` call performs the full read-render cycle and appends a new `function_call_output` item to the partition, even when an identical or subset span already exists in `_refreshable_items()` at the same revision and `status=live`.

Two problems follow from that:

1. **Context bloat.** Each redundant read adds a fresh live-window item — for a 200-line file that is roughly 200 numbered lines of duplicated content plus the `FILE ...` header. The reconciliation pass in `reconcile_tracked_files(...)` refreshes all of those windows in place on every later turn, so the duplication persists for the life of the conversation (or until compaction's `lift_active_live_windows` and live-window stripping steps strip it). Token cost is paid every round trip until then.
2. **No corrective feedback to the model.** Because the tool returns a normal `FILE ... status=live` view indistinguishable from a first-time read, the model has no signal that it just violated the guidance. The system-message rule is the only enforcement layer, and the trace in partition `17d85dc6-551c-40a4-a8d0-3562884647f4` shows it is not always sufficient — the model re-read `project/wip/context_loader/README.md` lines `1-200` three separate times (items 3, 38/40, and the failed item 32) and lines `201-255` twice (items 7 and 41/42), all within a single conversation and without any compaction in between.

The tool already has the state it would need to detect both cases. `_refreshable_items()` returns every existing live window with its `file_tool.path`, `file_tool.status`, `file_tool.view_start_line`, `file_tool.view_end_line`, `file_tool.revision`, and (when set) `file_tool.requested_end_line` annotations available on each item. Nothing on the read path consumes that information today.

## Proposed Fix

Before performing disk I/O in `_do_read_lines`, scan `_refreshable_items()` for an existing live window over the same path whose **intended coverage** fully covers the requested span. If one exists, return a short `REDUNDANT_READ` diagnostic that points the model at the existing window's coordinates and skips the disk read entirely.

Implementation outline:

1. Resolve the requested span:
   - Exact-span calls: `[start_line, end_line]`.
   - Open-ended pages (no `end_line`): `[start_line, start_line + max_lines - 1]`.

2. For each candidate window in `_refreshable_items()`, compute its **intended coverage end**:
   - If `file_tool.requested_end_line` is set on the window's annotations, use it. This covers exact-span reads where the request bounded the window, and `RANGE_TRUNCATED` items where the per-call cap bounded the window.
   - Otherwise use `view_start_line + max_lines - 1`. This handles open-ended reads where `view_end_line` was clipped at EOF: a 3-line file read open-ended from `start=1` has `view_end_line=3` but the harness's intended coverage was `[1, 200]`, and a follow-up open-ended re-read of `start=1` should still short-circuit.

   `view_end_line` alone is the wrong lookup key for open-ended reads — `render_view` (file_tool.py:938-961) clips it to the file's current line count, so it understates coverage whenever the file is shorter than the max page size. The 201-255 duplicate in the partition trace is likely an open-ended page that hit EOF.

3. Filter candidates to items with `file_tool.path == resolved_path` and `file_tool.status == "live"`, and to items that are **stable coverage** (see below).

4. If any candidate satisfies both `window.view_start_line <= requested_start_line` and `intended_coverage_end >= requested_end_line`, build and return a `REDUNDANT_READ` item before any filesystem call.

5. The `REDUNDANT_READ` item is not a live window. It carries no `file_tool.status=live`, no `file_tool.revision`, and no rendered file body — only a short textual body so the model spends a handful of tokens on it rather than re-rendering the file. Keep `file_tool.path` on the item so compaction's path-activity check in `_lift_active_live_windows` (web_v1/__init__.py:240-249) still recognizes the path as active and lifts the underlying live window into the new tail when needed.

### Stable-coverage classification

`_has_transient_file_diagnostic` (file_tool.py:1095-1102) currently bundles two distinct properties onto the same predicate:

- **"Re-render on refresh even at the same revision."** Used at file_tool.py:1058 so that ALREADY_EXISTS, CONFLICT, RANGE_ERROR, and RANGE_TRUNCATED items shed their diagnostic header on the next refresh and become clean live windows.
- **"Don't trust as stable coverage."** Proposed here for the redundant-read check.

These are not the same property. `RANGE_TRUNCATED` carries a real live window over `[view_start_line, view_end_line]` with `requested_end_line = cap_end_line` set; its diagnostic header is about the *request* having been over-cap, not about file state. After the model has seen one `RANGE_TRUNCATED` view, an immediate re-read of the truncated span is exactly the misuse the fix targets, so `RANGE_TRUNCATED` should count as stable coverage. `ALREADY_EXISTS` / `CONFLICT` / `RANGE_ERROR` carry decision-relevant state the model still needs to react to (and are write/create-side diagnostics, not read-side), so they should not count as coverage.

Either split `_has_transient_file_diagnostic` into `should_force_refresh` (all four) and `is_stable_coverage` (excludes only the first three), or add a new predicate alongside it. Both call sites should then use the predicate that matches their actual semantics.

### Output shape

```text
REDUNDANT_READ path=/workspace/context_loader/README.md requested_lines=1-200
An existing live window already covers this span (rendered lines 1-200, intended coverage 1-200, revision 0765de1927...). Use that window. To extend coverage, page forward from start_line=201.
```

Report both the rendered range (what is materialized in the window's body) and the intended coverage range (what the harness will refuse to re-read). For windows where `view_end_line == intended_coverage_end` (the common case) the two are identical. When the window was clipped at EOF — for example, an open-ended read of a 3-line file renders lines 1-3 but the harness's intended coverage is still 1-200 — the two ranges diverge, and the model needs both: rendered lines tell it what content it can rely on, intended coverage tells it where to page from (`intended_coverage_end + 1`) to escape the redundant-read check. The paging hint must point at `intended_coverage_end + 1`, **not** `view_end_line + 1` — in the 3-line-file case `view_end_line + 1 = 4` is still inside intended coverage and would either re-trigger `REDUNDANT_READ` or (for a span small enough to fit under coverage) trigger it on the spot.

Update the matching `system_message_parts` bullet so the model knows the tool will now actively reject fully-covered reads with a `REDUNDANT_READ` diagnostic, that the recovery is to consult the existing live window rather than reissue the read, and that `REDUNDANT_READ` means the tool did not consult disk (see "Accepted: same-turn freshness loss" below).

### Tradeoffs

This short-circuits a read the model intentionally asked for, which is a small departure from pure tool literalism. The two main alternatives are:

- **Always perform the read but add a warning note to its output.** That gives the model a corrective signal but does nothing about the context-bloat half of the problem — every redundant read still appends a fresh duplicate live window to the partition.
- **Always perform the read and dedup live windows for the same path at the partition level.** Cleaner long-term, but it touches reconciliation, compaction lift, and prefix-comparison logic in `prokaryotes/web_v1/__init__.py`. Out of scope for this issue; the short-circuit fix can ship first and a later cleanup can decide whether full-coverage dedup belongs on the read path or on the partition.

I prefer detecting only **full coverage** in v1 — the request fits entirely inside one existing window — rather than chasing partial overlaps, multi-window stitched coverage, or revision mismatches. Partial-coverage cases are rarer and the policy questions are messier: do we trim the request to the uncovered tail? do we return a hybrid view that combines several live windows? Starting with full-coverage detection captures the observed misuse pattern from the partition trace and leaves room to relax the check later.

**Accepted: same-turn freshness loss.** Today every `read_lines` call re-reads disk, computes a fresh revision, and calls `_refresh_live_windows` so all existing live windows for the path move to that revision. The reconcile pass at request start (`reconcile_tracked_files`) already keeps live windows current at *turn* boundaries, but a same-turn `shell_command` write (or any external write after request start) is currently caught only by a re-read. After this fix, a fully-covered re-read short-circuits without re-reading, so a model that mutated the file via `shell_command` within the same round and immediately re-issued `read_lines` would see stale content until the next turn's reconcile. The system message already steers the model away from `shell_command` for routine file edits, and `file_tool` write actions refresh live windows themselves, so the residual exposure is narrow — accepted in v1. Documenting this in the system-message bullet ("`REDUNDANT_READ` means the tool did not consult disk; if you need a fresh read, change the requested span or wait for the next turn's reconcile") makes the tradeoff legible to the model.

Two cases worth leaving as ordinary reads even when an existing window appears to cover the span:

- The existing window is a tombstone (`file_tool.status == "stale"`). The model may be trying to recover; let the read proceed and the harness will re-evaluate accessibility, either repromoting the path to a live window on success or refreshing the tombstone with the new error class on failure.
- The existing window is an `ALREADY_EXISTS`, `CONFLICT`, or `RANGE_ERROR` diagnostic. Those headers carry decision-relevant state the model still needs to react to, and they normalize to clean live windows on the next refresh. (`RANGE_TRUNCATED` is *not* in this group — see "Stable-coverage classification" above.)

Both conditions are cheap to evaluate against existing annotations and helpers — no new metadata is required.

### Test deltas

Two existing tests assume `read_lines` always touches disk and will need adjustment:

- `tests/unit_tests/test_file_tool.py::test_read_refreshes_prior_live_windows_for_same_path` — exercises the re-read refresh side effect. With the fix, a fully-covered re-read is short-circuited and does not refresh. The reconcile pass is the authoritative refresh path between turns, so this test should either use a **non-covered** read (a span that extends past the existing window's `intended_coverage_end` — for example, first read an exact `[1, 50]` span, then read `[1, 100]`, so the second read genuinely hits disk; "partially overlapping" is not enough because under intended coverage many partial overlaps are still fully covered and will short-circuit), or be split into a "reconcile refreshes" assertion and a separate "read short-circuits when covered" assertion.
- `tests/unit_tests/test_file_tool.py::test_failed_read_tombstones_prior_live_windows_for_same_path` — exercises the failed-read tombstone path. With the fix, if the existing window is live and fully covers the requested span, the tool short-circuits before discovering the file became inaccessible mid-turn. Tombstone discovery for the covered case shifts entirely to the per-turn reconcile pass. Update the test to either trigger the failure on a non-covered span or move the assertion onto `reconcile_tracked_files`.

New tests to add:

- Fully-covered exact-span re-read returns `REDUNDANT_READ` and does not append a new live window.
- Fully-covered open-ended re-read of a short file (`view_end_line < view_start_line + max_lines - 1` because of EOF clipping) returns `REDUNDANT_READ`.
- Open-ended re-read of the next page (`start_line = intended_coverage_end + 1`) does *not* short-circuit. Use a setup where `view_end_line < intended_coverage_end` (open-ended first read of a short file) to verify the test would fail if the implementation used `view_end_line + 1` instead.
- `RANGE_TRUNCATED` window counts as stable coverage for its `[view_start_line, view_end_line]` span.
- `ALREADY_EXISTS` / `CONFLICT` / `RANGE_ERROR` windows do *not* count as stable coverage; the read proceeds.
- `REDUNDANT_READ` item carries `file_tool.path` but no `file_tool.status` / `file_tool.revision`, so `_lift_active_live_windows` still treats the path as active while `_refresh_live_windows` and `_tombstone_live_windows` skip it.

## Second Opinion

**Verdict:** Valid; the original Proposed Fix was incomplete, and has been revised in place.

The factual claims — that `_do_read_lines` always hits disk, that `_refreshable_items()` already surfaces the state needed to detect redundancy, that the system-message rule is currently the only enforcement layer — all check out against the code. The partition trace and the system-message text match the cited code paths. Line citations were stale at write-time (`86-138` → `86-179`, `248-254` → `289-295`, `497-503` → `551-557`); fixed in the Location section.

**Corrected: coverage lookup key was wrong for open-ended pages.**

The original proposal computed each candidate window's coverage as `[view_start_line, view_end_line]`. But `render_view` (file_tool.py:938-961) caps `view_end_line` at the file's current line count, so an open-ended read of a 3-line file produces `view_end_line=3`. A follow-up open-ended re-read of `start=1` then maps to a requested span of `[1, 200]`, which `[1, 3]` does not cover, and the redundant read goes through. The 201-EOF duplicate in the partition trace is the same shape: an open-ended page whose `view_end_line` was clipped at EOF.

The revised Proposed Fix uses **intended coverage end**: `file_tool.requested_end_line` when annotated, else `view_start_line + max_lines - 1`. That correctly identifies an open-ended read of a short file as covering `[start, start + max_lines - 1]`. Cross-checked against `_refresh_live_windows` (file_tool.py:1064-1070), which uses the same fallback (`max_lines` when `requested_end_line` is absent), so the lookup key and the refresh path agree on what each window's intended span is.

**Corrected: `RANGE_TRUNCATED` was wrongly excluded from coverage.**

The original proposal reused `_has_transient_file_diagnostic` to filter unstable items, lumping `RANGE_TRUNCATED` together with `ALREADY_EXISTS` / `CONFLICT` / `RANGE_ERROR`. But `_has_transient_file_diagnostic` actually encodes "re-render on refresh even at the same revision" (so the diagnostic header gets dropped on the next refresh) — that property is orthogonal to whether the window is usable as coverage. `RANGE_TRUNCATED` carries a real live window with `view_end_line = cap_end_line` and `requested_end_line = cap_end_line` set; its diagnostic header is about the *request*, not about file state. Excluding it lets a model that just got back `RANGE_TRUNCATED` for `[1, 250]` immediately re-issue `[1, 200]` and pay the full read cost — exactly the misuse the fix is supposed to catch.

The revised Proposed Fix splits the predicate: only ALREADY_EXISTS / CONFLICT / RANGE_ERROR are excluded from coverage; RANGE_TRUNCATED counts as stable coverage for its returned span. The split also leaves the existing refresh behavior (drop the diagnostic header on next refresh) untouched.

**Sharpened: same-turn freshness tradeoff.**

The original tradeoff section did not name the freshness loss explicitly. Every current `read_lines` call re-reads disk and calls `_refresh_live_windows`. After this fix, a fully-covered re-read skips both. The per-turn reconcile pass keeps live windows current at *turn* boundaries, so the residual exposure is a same-turn external write — most plausibly the model writing through `shell_command` and immediately re-reading. The system message already discourages that pattern and `file_tool` writes refresh windows themselves. Accepted; called out in the Proposed Fix's "Accepted: same-turn freshness loss" subsection, and the system-message update spells out that `REDUNDANT_READ` means the tool did not consult disk so the model knows when to alter the requested span instead.

**Added: test deltas.**

The original Proposed Fix did not flag that `test_read_refreshes_prior_live_windows_for_same_path` and `test_failed_read_tombstones_prior_live_windows_for_same_path` both assume `read_lines` unconditionally touches disk and will fail after the fix. Both are now called out, along with a starter list of new tests that should land alongside the fix.

**Corrected (follow-up pass): diagnostic paging hint and test descriptions had not been updated to match the intended-coverage model.**

A follow-up review caught that the Proposed Fix still phrased the paging hint as "`view_end_line + 1`" in the `REDUNDANT_READ` output shape, and used "partially-overlapping read" / "`start_line = view_end_line + 1`" in the test deltas. Both are inconsistent with intended coverage: in the EOF-clipped case (3-line file, intended coverage 1-200), `view_end_line + 1 = 4` is still inside the harness's coverage region, and a "partially overlapping" follow-up read can be entirely covered and short-circuit. Diagnostic output now reports both rendered range and intended coverage and pages from `intended_coverage_end + 1`; the test delta for the existing refresh test now says "non-covered read" with a concrete `[1, 50]` then `[1, 100]` setup; the next-page short-circuit test now uses `intended_coverage_end + 1` and recommends an EOF-clipped first read so the test would catch a mistakenly-used `view_end_line + 1` implementation.

**Recommendation adjustment**

Implement the revised Proposed Fix. Folded in: severity tag (Medium), corrected line citations, intended-coverage-end lookup, RANGE_TRUNCATED reclassified as stable coverage, explicit same-turn freshness tradeoff, test-delta notes for the two existing tests plus a new-test starter list, and the follow-up corrections that align the diagnostic output and test descriptions with the intended-coverage model.
