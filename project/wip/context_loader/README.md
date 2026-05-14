# Automatic Context File Discovery Design

## Goals

Replace the earlier standalone `context_loader` tool idea with automatic harness-managed context-file discovery.

The feature should:
- run automatically on every web-harness turn
- detect when new relevant workspace paths enter the current conversation context
- walk upward from those paths toward a configured boundary
- discover local context files named `CLAUDE.md`, `AGENTS.md`, and `README.md`
- surface discovered file locations in the system/developer prompt so the model knows they exist
- clearly annotate symlink alias relationships so the model can tell when multiple paths lead to the same underlying file
- make it easy for the model to use `file_tool.read_lines` to inspect discovered files when needed
- avoid overengineering ranking or retrieval logic

This design intentionally does **not** add a new tool for reading context files. Context files are ordinary files, and actual file contents should continue to enter the model-visible context through `file_tool`.

## Observed repository context

Observed in the current repository:
- `project/features/file_tool/README.md` defines `file_tool` as the structured way for the model to read files, with live-window refresh semantics and compaction-aware behavior.
- `prokaryotes/tools_v1/README.md` documents reusable `FunctionToolCallback` tools. Existing tools are `FileTool`, `ThinkTool`, and `ShellCommandTool`.
- `prokaryotes/harness_v1/web.py` builds a per-turn instruction message after `sync_context_partition(...)` and `reconcile_tracked_files(...)`; `impl="openai"` uses a developer message and `impl="anthropic"` uses a system message.
- The web harness currently registers `FileTool`, `ThinkTool`, and `ShellCommandTool`; it does not expose a context-loader tool.
- `prokaryotes/api_v1/models.py` provides `ContextPartition`, `ContextPartitionItem`, and internal `prokaryotes_annotations` metadata.
- Symlink aliases already exist in the repo, including:
  - `/app/project/wip/CLAUDE.md -> README.md`
  - `/app/project/wip/AGENTS.md -> README.md`
  - `/app/project/issues/CLAUDE.md -> README.md`
  - `/app/project/issues/AGENTS.md -> README.md`
  - `/app/prokaryotes/tools_v1/CLAUDE.md -> README.md`
  - `/app/prokaryotes/tools_v1/AGENTS.md -> README.md`
- These examples confirm that alias relationships are common enough that the prompt should surface them clearly.

## Core concepts and new abstractions

### Harness-managed context discovery

Context-file discovery becomes a harness responsibility rather than a model-invoked tool.

Responsibilities:
- collect candidate workspace paths relevant to the current turn
- detect newly entered paths conservatively
- walk upward from those paths toward the workspace-root boundary
- inspect candidate context filenames in each directory
- group discovered paths by underlying real file for ranking purposes
- preserve all matched paths for display purposes
- render a compact prompt section describing discovered files and alias relationships

### Candidate path sources

Initial path sources should be conservative.

High-confidence sources:
- `file_tool` paths already present in the active `ContextPartition`
- explicit workspace-relative or absolute path mentions in user messages
- explicit path arguments from structured tool calls or tool metadata when they parse cleanly inside the workspace

Lower-confidence sources that should remain out of scope for v1:
- raw `file_tool` live-window bodies
- arbitrary assistant prose that happens to contain slash-delimited text
- unstructured tool output bodies unless the tool defines a stable path field or path annotation

The first version should prefer missing some valid paths over surfacing many weak false positives.

### Directory-level candidate selection

For each directory in the upward walk, inspect these filenames:
- `CLAUDE.md`
- `AGENTS.md`
- `README.md`

For each existing candidate, capture:
- matched path
- matched filename
- whether it is a symlink
- resolved real path
- whether it is a regular file
- directory distance from the originating relevant path
- the originating path or paths that led to discovery

Selection rules for the initial version:
- do not reject candidates simply because they are symlinks
- resolve each candidate to its real path
- after resolution, require the real path to remain inside `workspace_root`
- skip candidates that are not regular files
- preserve all matched paths in the displayed result set
- group candidates by resolved real path for ranking so one underlying file does not consume multiple ranking slots

This keeps alias visibility for the model while still preventing alias-heavy directories from dominating ranking.

### Discovery groups vs displayed matches

The feature should maintain two views of the same discovery result:

1. **Display view**
   - every matched path is surfaced
   - symlink relationships are shown explicitly
   - if multiple names in one directory resolve to the same file, the model can see that directly

2. **Ranking view**
   - matches that resolve to the same real file belong to one group
   - the group receives one score
   - the rendered prompt may show one ranked group with all of its alias paths nested underneath

### Ranking philosophy

Ranking exists only to keep the prompt compact and practical.

The initial heuristic should remain simple and transparent:
- stronger path source beats weaker path source
- more recent path activity beats older path activity
- more frequent mentions beat rarer mentions
- nearer ancestor files beat farther ancestor files

The target is not perfect retrieval. The goal is only to make it a little easier for the model to notice likely relevant local guidance.

## Traversal and boundary detection

The upward search should follow these rules:
- if the relevant path is a file, start from its parent directory
- if the relevant path is a directory, start from that directory
- walk upward toward the workspace root
- stop at the workspace root in the initial version

Initial boundary behavior:
- no repository-root detection in v1
- no caller-provided override in v1, because this is harness-managed rather than tool-invoked

## Data model changes

No provider-facing schema changes are required for the initial version.

Potential internal metadata additions may be useful, but are not strictly required to land v1. Likely examples:
- discovered-path source strength
- first-seen / last-seen turn bookkeeping
- mention counts for ranking
- cached discovery groups for prompt rendering

Do **not** attach discovery metadata to ordinary user/assistant message items in `ContextPartition.items`, because partition sync compares those message items against raw `ChatConversation` items by equality. If persistence is needed, prefer partition-level cache state or annotations on non-message internal items only, with regression tests covering sync behavior.

## Protocol changes

### No new tool

The revised design does **not** add a `context_loader` tool.

There is no new:
- `FunctionToolCallback`
- tool registration entry
- tool schema
- function-call output shape for context loading

### Prompt injection behavior

Instead, the web harness appends a compact context-discovery section to the per-turn system/developer prompt.

That section should:
- state that relevant local context files were detected near paths in the conversation
- list discovered paths, including symlink annotations
- render paths and provenance using escaped or structured serialization so filenames cannot inject prompt text
- present the highest-ranked groups first
- indicate when additional lower-ranked groups were omitted from detailed display
- instruct the model to use `file_tool.read_lines` if it wants to inspect any file contents
- explicitly tell the model not to assume file contents without reading them

### Prompt shape

Initial rendering shape:
- short heading
- compact ranked list of up to 10 groups
- for each group, list all matched paths and alias notes
- optional brief provenance note such as which relevant path caused the discovery
- final instruction reminding the model to use `file_tool.read_lines`

Example shape:

```text
# Local context files detected

The following files were found near paths relevant to this turn. They may contain local guidance. Use `file_tool.read_lines` to inspect them if useful. Do not assume their contents without reading them.

1. Context group: {"real_path": "/app/project/wip/README.md"}
   - {"path": "/app/project/wip/README.md", "kind": "regular"}
   - {"path": "/app/project/wip/CLAUDE.md", "kind": "symlink", "target": "/app/project/wip/README.md"}
   - {"path": "/app/project/wip/AGENTS.md", "kind": "symlink", "target": "/app/project/wip/README.md"}
   - discovered_from: {"path": "/app/project/wip/context_loader/README.md"}

2. Context group: {"real_path": "/app/project/README.md"}
   - {"path": "/app/project/README.md", "kind": "regular"}

Additional lower-ranked context groups omitted: 3
```

## Redesigned or new functions with pseudocode

The earlier `ContextLoaderTool.call(...)` design is replaced by harness-side discovery helpers.

### Per-turn integration in the web harness

```python
async def post_chat(...):
    context_partition = await self.sync_context_partition(conversation)
    workspace_root = Path.cwd()
    await reconcile_tracked_files(context_partition, workspace_root=workspace_root)

    file_tool = FileTool(context_partition, workspace_root=workspace_root)
    shell_command_tool = ShellCommandTool()
    think_tool = ThinkTool(self.llm_client, model)

    discovered_context = discover_relevant_context_files(
        context_partition=context_partition,
        workspace_root=workspace_root,
    )
    discovery_prompt_lines = render_context_discovery_prompt(discovered_context)

    message_parts = []
    message_parts.extend(core_instruction_parts)
    message_parts.extend(runtime_parts)
    message_parts.extend(tool_usage_parts)
    if discovery_prompt_lines:
        message_parts.append("")
        message_parts.extend(discovery_prompt_lines)
    message_parts.extend(personality_parts)
    message_parts.extend(user_context_parts)
```

### Helper: collect candidate paths

```python
def collect_candidate_paths(context_partition: ContextPartition, workspace_root: Path) -> list[PathCandidate]:
    candidates = []

    for item in context_partition.items:
        candidates.extend(paths_from_file_tool_annotations(item, workspace_root))
        candidates.extend(paths_from_user_message_text(item, workspace_root))
        candidates.extend(paths_from_structured_tool_fields(item, workspace_root))

    return merge_and_score_candidates(candidates)
```

### Helper: upward discovery

```python
def discover_relevant_context_files(
    context_partition: ContextPartition,
    workspace_root: Path,
) -> DiscoveryResult:
    candidates = collect_candidate_paths(context_partition, workspace_root)
    relevant_paths = select_new_or_high_value_paths(candidates)

    matches = []
    resolved_workspace_root = workspace_root.resolve()
    for candidate in relevant_paths:
        resolved_candidate_path = candidate.path.resolve()
        try:
            resolved_candidate_path.relative_to(resolved_workspace_root)
        except ValueError:
            continue

        start_dir = resolved_candidate_path if resolved_candidate_path.is_dir() else resolved_candidate_path.parent
        for directory in iter_dirs_upward(start_dir, stop_at=resolved_workspace_root):
            for filename in ["CLAUDE.md", "AGENTS.md", "README.md"]:
                matched_path = directory / filename
                if not matched_path.exists():
                    continue
                resolved_path = matched_path.resolve()
                try:
                    resolved_path.relative_to(resolved_workspace_root)
                except ValueError:
                    continue
                if not resolved_path.is_file():
                    continue
                matches.append(build_match(candidate, matched_path, resolved_path=resolved_path))

    groups = group_matches_by_real_path(matches)
    ranked_groups = rank_groups(groups)
    return DiscoveryResult(matches=matches, groups=ranked_groups)
```

### Helper: upward directory walk

```python
def iter_dirs_upward(start_dir: Path, stop_at: Path) -> list[Path]:
    dirs = []
    current = start_dir
    while True:
        dirs.append(current)
        if current == stop_at:
            break
        if current.parent == current:
            break
        current = current.parent
    return dirs
```

### Helper: ranking

```python
def rank_groups(groups: list[DiscoveryGroup]) -> list[DiscoveryGroup]:
    for group in groups:
        group.score = (
            score_source_strength(group)
            + score_recency(group)
            + score_frequency(group)
            - score_distance_penalty(group)
        )
    return sorted(groups, key=lambda group: group.score, reverse=True)
```

### Helper: prompt rendering

```python
def render_context_discovery_prompt(result: DiscoveryResult, max_groups: int = 10) -> list[str]:
    if not result.groups:
        return []

    lines = [
        "# Local context files detected",
        "",
        "The following files were found near paths relevant to this turn. "
        "They may contain local guidance. Use `file_tool.read_lines` to inspect them if useful. "
        "Do not assume their contents without reading them.",
    ]

    for idx, group in enumerate(result.groups[:max_groups], start=1):
        lines.append("")
        lines.append(f"{idx}. Context group: {json.dumps({'real_path': str(group.real_path)})}")
        for match in group.matches:
            lines.append(render_match_line(match))

    omitted = len(result.groups) - max_groups
    if omitted > 0:
        lines.append("")
        lines.append(f"Additional lower-ranked context groups omitted: {omitted}")

    return lines
```

## Infrastructure changes

Initial implementation files are likely to include:
- `prokaryotes/harness_v1/web.py`
- a new shared helper module for path extraction, upward discovery, grouping, ranking, and prompt rendering

Possible review targets before implementation:
- `prokaryotes/api_v1/models.py` if lightweight internal metadata is needed
- `prokaryotes/web_v1/__init__.py` if shared harness-level helpers belong there
- tests covering both provider harnesses and prompt construction behavior

## Testing

The implementation should add or update tests at three levels.

### Unit tests

Add focused unit coverage for the shared discovery helpers.

Primary areas:
- extracting conservative path candidates from `ContextPartitionItem` message text and `file_tool` annotations
- not producing candidates from raw `file_tool` live-window body text
- not producing candidates from arbitrary assistant prose containing slash-delimited text
- not producing candidates from unstructured tool output bodies that lack a stable path field or path annotation
- walking upward from file and directory paths to the workspace-root boundary
- discovering `CLAUDE.md`, `AGENTS.md`, and `README.md` candidates
- rejecting symlinked context-file candidates whose resolved real paths escape `workspace_root`
- preserving all matched paths while grouping by resolved real path for ranking
- rendering explicit symlink annotations in the prompt section
- rendering paths and provenance with escaped or structured serialization, including malicious filenames containing newlines, markdown, or prompt-looking text
- ranking by simple source-strength / recency / frequency / distance heuristics
- truncating prompt rendering to the top 10 ranked groups and reporting omitted groups
- producing no prompt section when no relevant context files are found

Likely files:
- a new unit test module for the discovery helper implementation
- `tests/unit_tests/test_openai_v1.py` if prompt-construction assertions are added at the provider client boundary
- `tests/unit_tests/test_anthropic_v1.py` if provider message-shape assertions are added there
- `tests/unit_tests/test_system_message_utils.py` only if shared prompt-rendering helpers are added in that module

### Integration tests

Add Tier B integration coverage that exercises the real web stack for both providers.

Primary scenarios:
- a user message mentions a workspace path and the next turn's system/developer prompt includes discovered local context-file locations
- a prior `file_tool.read_lines` path causes nearby context files to be surfaced on a later turn
- multiple alias paths in one directory are all surfaced with clear symlink-target annotations
- multiple matched filenames resolving to one underlying real file occupy one ranking group rather than multiple top-level ranking slots
- nearest ancestor files rank above farther ancestor files when both are discovered from the same relevant path
- path mentions outside the workspace or malformed path-like strings are ignored conservatively
- when more than 10 ranked groups exist, only the top 10 groups are expanded and the omitted count is shown
- when no files are discovered, no extra context-discovery prompt section is injected
- behavior is consistent across `WebHarness(impl="openai")` and `WebHarness(impl="anthropic")`

Likely files:
- `tests/integration_tests/tier_b/test_chat_flow.py`
- a new Tier B integration test module dedicated to automatic context-file discovery
- `tests/integration_tests/tier_b/test_file_tool_flow.py` if reusing existing file-tool-driven multi-turn patterns is useful

### Regression and compaction-sensitive coverage

Because the feature changes per-turn prompt construction, add regression tests around prompt stability and interaction with existing tracked-context behavior.

Primary scenarios:
- discovered file-location hints do not require any new tool registration or tool-call protocol changes
- the injected discovery section coexists with existing tool-usage guidance and user-context sections in the correct order
- compaction and later turns continue to rebuild the discovery section from current context rather than relying on stale serialized prompt text

Likely files:
- `tests/unit_tests/test_web_v1.py`
- `tests/unit_tests/test_openai_v1.py`
- `tests/unit_tests/test_anthropic_v1.py`
- `tests/unit_tests/test_compaction_swap.py` or a new targeted compaction regression test if prompt reconstruction needs explicit coverage

### Verification target

Before implementation is considered complete, the expected regression suite should include the relevant new unit and integration tests plus the existing harness and file-tool suites that cover adjacent behavior.
## Open questions

1. Should harness-side discovery read any file contents directly, or should v1 surface locations only and rely entirely on later `file_tool.read_lines` calls for content?
   - Current proposal: surface locations only.
2. Should discovery run over all relevant paths every turn, or only newly entered paths plus cached prior hits?
   - Current proposal: start by emphasizing newly entered paths, with simple reuse of already-known hits as needed for stable prompt rendering.
3. Exactly what text patterns should count as conservative path mentions?
   - Current proposal: explicit relative/absolute paths that parse cleanly inside the workspace.
4. How should prompt rendering distinguish between ranked groups and complete displayed matches most compactly?
   - Current proposal: rank groups, but display all matched paths nested under each shown group.
5. Should hidden variants such as `.claude.md` be considered later?
   - Current proposal: no, not in the initial version.
6. Should repository-root detection replace workspace-root stopping later?
   - Current proposal: maybe later, but not in the first version.

## Initial implementation notes

The first version should optimize for:
- transparency
- predictable traversal
- low policy complexity
- low token overhead
- reuse of existing `file_tool` behavior for actual file reading
- consistent behavior across both web harnesses
- faithful support for symlink-alias workflows already present in the repository

The first version should not attempt to:
- add a new context-loading tool
- auto-read all discovered file contents into the prompt
- infer semantic relevance beyond simple heuristics
- interpret or merge instructions semantically
- discover arbitrary documentation files beyond the three named candidates
- build a complex retrieval or ranking system
