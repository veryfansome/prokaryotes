# Context Loader

## Overview

On every web-harness turn, the harness walks upward from each workspace path that is currently relevant to the conversation, collects any `CLAUDE.md` / `AGENTS.md` / `README.md` it finds, and surfaces those file *locations* in the per-turn system/developer prompt. Symlink relationships are spelled out so the model can tell when multiple paths resolve to the same underlying file, and groups are ranked so the prompt stays compact even when many candidates each find many context files.

This is not a tool. Context files are ordinary files; `file_tool.read_lines` remains the only path for actual file contents to enter the model-visible context. Discovery surfaces *what exists nearby*, not *what's in it*.

Wired into `WebHarness`; `SlackHarness` is not yet wired but the helper is harness-agnostic.

---

## Candidate Path Sources

Three sources contribute candidate paths, in decreasing strength. Each path is resolved to an absolute path before deduplication.

### Live working-file windows (strongest)

Each entry in `conversation.working_file_windows` with `status == "live"` and `source_kind != "tombstone"` contributes its `path`. Tombstoned/stale entries survive in the list after a path disappears or becomes unreadable (see `live_windows.tombstone_windows_for_path()`) and are skipped.

### Surviving `file_tool.path` annotations (medium)

Every `function_call_output` `TurnItem` in `historical_turns.values()` carrying a `file_tool.path` annotation contributes its path. `historical_turns` is keyed on bot messages still present in `conversation.messages`, so compacted-out turns don't contribute.

Two filtering rules matter:

- **Drop annotations whose path no longer exists on disk.** The annotation is durable across call shapes (including `REDUNDANT_READ` and history-only outputs), so a deleted path would keep surfacing nearby context docs until the bot message compacts away.
- **Do not gate on live-window membership.** Successful `create_file` and edit calls stamp the annotation without minting a window; a live-window gate would suppress discovery for files the model just created or edited.

### Explicit path mentions in user messages (weakest)

Filter to `author_id != conversation.bot_author_id`. The acceptance rule:

1. Extract whitespace-delimited tokens from message text.
2. Reject tokens containing `://` (URLs).
3. Clean each token: strip surrounding backticks, parens, brackets; trailing punctuation (`.,;:?!)>]}`); and any trailing `:<digits>` or `:<digits>-<digits>` line-reference suffix. Punctuation and line-ref strips alternate until stable so either order works (`path/file:42,` and `path/file,:42` both reduce to `path/file`).
4. Resolve as `(workspace_root / token).resolve()` — Pathlib's `/` returns `token` unchanged when absolute and joins otherwise, so absolute, `./`-prefixed, and workspace-relative mentions all work.
5. Accept iff the cleaned token contains at least one `/` AND the resolved path exists on disk inside `workspace_root`.

The slash requirement keeps the rule conservative. Bare filenames (e.g. `README.md`, `pyproject.toml`) are rejected even when the file exists at the workspace root — those mentions are too ambiguous to ground reliably. Slash-bearing prose like `A/B` or `2026/05/25` resolves to non-existent paths and is rejected by the existence check. The `file_path:line_number` form is supported because the project's conventions reference code that way, and the path component always carries a slash.

### Excluded sources

Bot-authored content is excluded entirely. Structured tool arguments authored by the model (e.g. `ThinkTool.paths`) would create a self-reinforcing prompt-drift loop with no human or `file_tool` grounding. Raw `rendered_output` text from live windows, assistant prose containing slash-delimited text, and unstructured tool output bodies are also excluded — they lack a stable path field or annotation.

---

## Candidate Dedupe

One absolute path can appear in multiple sources within a single turn — most commonly when a single `read_lines` call mints both a `WorkingFileWindow` and a persisted `file_tool.path` annotation on its `function_call_output`. `collect_candidate_paths` collapses these so a single path produces a single candidate:

- Dedupe by resolved absolute path. One path produces one candidate.
- The retained candidate carries the **max** source strength across its contributing signals (live window > annotation > user mention).

Per-path mention frequency is intentionally not counted. Ranking uses **breadth** instead — the number of distinct origin candidates whose upward walks reach a group. Two `read_lines` of the same file across different turns still produce one candidate; the ranking signal comes from how many *different* originating paths reach the same group.

Distance is not a candidate property — it is a property of `(origin, match)` pairs and only exists after the upward walk.

---

## Upward Walk

For each candidate, the start directory is the candidate itself if it's a directory, otherwise its parent. From there the walk iterates parents up to and including `workspace_root`. Distance is the number of `parent` steps from start_dir to the matched file's parent (0 = same directory as the origin).

No repository-root detection and no caller-provided override — the workspace root is the right boundary for a harness-managed feature with no external caller.

---

## Match Resolution

At each directory in the walk, the harness inspects `CLAUDE.md`, `AGENTS.md`, and `README.md`. Each existing match produces a `DiscoveryMatch` carrying the matched path, the resolved real path, the kind (`regular` or `symlink`), and the distance.

Acceptance rules:

- Symlinks are allowed; resolve to the real path before accepting.
- After resolution, the real path must remain inside `workspace_root`. Cross-subtree aliases that resolve inside the workspace are fine (e.g. `project/ideas/think_tool/CLAUDE.md → ../../features/think_tool/README.md`); symlinks whose targets escape the workspace are rejected.
- The real path must be a regular file.

When two origins hit the same matched path (common when they share an ancestor — `prokaryotes/foo.py` and `prokaryotes/bar.py` both finding `prokaryotes/README.md`), the matches dedupe by `matched_path` within the group and the origins collapse into a `discovered_from_paths` list. No duplicate alias rows in the rendered prompt.

---

## Groups vs Displayed Matches

The feature maintains two views of the same discovery result:

1. **Display view** — every matched path is surfaced, symlink relationships are shown explicitly, and if multiple names in one directory resolve to the same file, the model sees that directly.
2. **Ranking view** — matches that resolve to the same real file belong to one group; the group receives one score; the rendered prompt shows one ranked group with all of its alias paths nested underneath.

Intra-group ordering is canonical so output is reproducible:

- alias rows: regular file first, then symlinks; ties broken by `matched_path` string ascending
- `discovered_from_paths` entries: absolute-path string ascending

---

## Ranking

Ranking exists only to keep the prompt compact. Scoring happens at the group level after discovery — ancestor distance is a property of an `(origin, match)` pair and does not exist until the walk runs.

Each match carries its origin's source strength and the directory distance from origin to match; `group_matches_by_real_path` aggregates those onto the group; `rank_groups` reads only the aggregates.

Lexicographic priority (no additive weights, no per-signal tunable):

1. **stronger source beats weaker** — max source strength across the group's matches.
2. **broader attention beats narrower** — `len(discovered_from_paths)`, the number of distinct origin candidates whose upward walks reached this group. Within-group `matched_path` dedupe prevents a single origin from inflating breadth by hitting the same `real_path` through several ancestor steps.
3. **nearer ancestor beats farther** — min directory distance from origin to match across the group's matches.
4. **deterministic tiebreak** — `real_path` ascending.

---

## Prompt Rendering

The web harness appends a compact context-discovery section to the per-turn system/developer prompt assembled in `WebHarness._build_instruction_parts`, between the `# Tool usage` block and the `# User context` block. The section:

- opens with a short heading and one-sentence framing that reminds the model to use `file_tool.read_lines` and not to assume contents
- lists up to 10 ranked groups (`max_groups`); for each group, all matched paths with explicit symlink annotations and up to 3 provenance origins (`max_provenance_per_group`)
- emits an `Additional lower-ranked context groups omitted: N` trailer when groups exceed `max_groups`, and a `{"omitted": N}` line in the same JSON shape when origins exceed `max_provenance_per_group`

Paths and provenance render via `json.dumps` so filenames cannot inject prompt text. Example with three live `read_lines` reads from `prokaryotes/tools_v1/file_tool/reads.py`, `prokaryotes/utils_v1/system_message_utils.py`, and `prokaryotes/web_v1/auth.py` — yielding four groups, two of which are three-origin (shown) and two of which are single-origin (omitted, demonstrating that breadth outranks distance):

```text
# Local context files detected

The following files were found near paths relevant to this turn. They may contain local guidance. Use `file_tool.read_lines` to inspect them if useful. Do not assume their contents without reading them.

1. Context group: {"real_path": "/app/prokaryotes/README.md"}
   - {"path": "/app/prokaryotes/README.md", "kind": "regular"}
   - {"path": "/app/prokaryotes/AGENTS.md", "kind": "symlink", "target": "/app/prokaryotes/README.md"}
   - {"path": "/app/prokaryotes/CLAUDE.md", "kind": "symlink", "target": "/app/prokaryotes/README.md"}
   - discovered_from: {"path": "/app/prokaryotes/tools_v1/file_tool/reads.py"}
   - discovered_from: {"path": "/app/prokaryotes/utils_v1/system_message_utils.py"}
   - discovered_from: {"path": "/app/prokaryotes/web_v1/auth.py"}

2. Context group: {"real_path": "/app/README.md"}
   - {"path": "/app/README.md", "kind": "regular"}
   - {"path": "/app/AGENTS.md", "kind": "symlink", "target": "/app/README.md"}
   - {"path": "/app/CLAUDE.md", "kind": "symlink", "target": "/app/README.md"}
   - discovered_from: {"path": "/app/prokaryotes/tools_v1/file_tool/reads.py"}
   - discovered_from: {"path": "/app/prokaryotes/utils_v1/system_message_utils.py"}
   - discovered_from: {"path": "/app/prokaryotes/web_v1/auth.py"}

Additional lower-ranked context groups omitted: 2
```

### Prompt cache impact

The discovery section sits inside the per-turn system/developer prompt and varies turn-to-turn as `working_file_windows`, annotations, and user mentions shift. Any prompt-cache breakpoint set after `# Tool usage` is invalidated each turn. The cost is accepted — the section is small (≤10 groups of short structured lines).

---

## Design Notes

- **No new data model.** Discovery recomputes from current `Conversation` state every turn; no fields are added to `Conversation`, `WorkingFileWindow`, `TurnItem`, or `ConversationMessage`. If a later increment needs to persist discovery metadata across turns, prefer a new field on `Conversation` (sibling to `working_file_windows`) over reusing `TurnItem.prokaryotes_annotations`, so `ConversationSyncer`'s Redis/ES reconciliation doesn't have to learn about discovery state.
- **Recency is not a ranking signal.** `WorkingFileWindow` has no turn/timestamp field, and compaction carries `working_file_windows` forward through an `origin_call_ids` filter without preserving original turn position. Adding recency would require a schema change; ranking by source strength + breadth + distance was deemed enough.
- **Per-path mention frequency is not a ranking signal.** Breadth (distinct originating paths reaching a group) is used instead. Revisit if a single hot path consistently outranks broader signals.
- **Hidden variants like `.claude.md` are not recognized.** Only `CLAUDE.md`, `AGENTS.md`, `README.md`.
- **Workspace root is the only boundary.** No git-root or `.contextroot` detection.

---

## Relationship to the codebase

- `prokaryotes/utils_v1/context_discovery.py` — the implementation. Public surface: `discover_relevant_context_files`, `render_context_discovery_prompt`, plus the `PathCandidate` / `DiscoveryMatch` / `DiscoveryGroup` dataclasses and the per-source `paths_from_*` leaves. `SOURCE_RANK` is module-level so the leaves, the grouper, and `rank_groups` all share one source of truth.
- `prokaryotes/harness_v1/web.py` — `WebHarness._build_instruction_parts` calls `discover_relevant_context_files` then `render_context_discovery_prompt`, slotting the section between `# Tool usage` and `# User context`. `_dispatch_turn` threads `historical_turns` and `workspace_root` through.
- `tests/unit_tests/test_context_discovery.py` — covers candidate extraction from each source, dedupe across sources, the upward walk, symlink rules, grouping, ranking, and rendering.
- `tests/unit_tests/test_web_harness_instruction.py::TestDiscoverySection` — instruction-parts assembly: absence when no candidates match, presence when a live window neighbors a README, and ordering between `# Tool usage` and `# User context`.
- `tests/integration_tests/tier_b/test_context_discovery_flow.py` — end-to-end across the real Redis/Postgres/Elasticsearch stack for both providers: validates that a user path mention surfaces in the next turn's instruction, and that a prior turn's `file_tool.path` annotation does too (via `_dispatch_turn`'s historical-turns lookup).
