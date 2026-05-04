# Context Loader Design

## Goals

Add a reusable `context_loader` tool under `prokaryotes/tools_v1/` that loads path-local guidance files for the assistant.

The tool should:
- accept a file or directory path
- walk upward from that path toward a configured boundary
- discover local context files named `CLAUDE.md`, `AGENTS.md`, and `README.md`
- avoid redundant loading when `CLAUDE.md` and `AGENTS.md` are symlinks to `README.md`
- preserve intentionally distinct files when multiple context files in the same directory are different
- return loaded context in a transparent, structured format so the model can incorporate it into later reasoning

The tool is intended to support system/developer prompts that instruct the model to consider loading local context before planning, editing, or reviewing code in a project subtree.

## Observed repository context

Observed in the current repository:
- `prokaryotes/tools_v1/README.md` defines reusable `FunctionToolCallback` implementations and documents the required tool contract.
- `prokaryotes/tools_v1/` currently contains `shell_command.py`, `think.py`, and an empty `__init__.py`.
- Symlink aliases already exist in the repo, including:
  - `/app/project/wip/CLAUDE.md -> README.md`
  - `/app/project/wip/AGENTS.md -> README.md`
  - `/app/project/issues/CLAUDE.md -> README.md`
  - `/app/project/issues/AGENTS.md -> README.md`
  - `/app/prokaryotes/tools_v1/CLAUDE.md -> README.md`
  - `/app/prokaryotes/tools_v1/AGENTS.md -> README.md`
- These examples confirm that duplicate-loading from alias filenames is a real implementation concern.

## Core concepts and new abstractions

### `ContextLoaderTool`

A new `FunctionToolCallback` implementation in `prokaryotes/tools_v1/context_loader.py`.

Responsibilities:
- parse tool arguments
- resolve the starting directory from the provided path
- walk upward through parent directories
- inspect candidate context filenames in each directory
- deduplicate redundant files by underlying file identity
- read and return the contents of selected files
- expose tool usage guidance through `system_message_parts`

## Directory-level candidate selection

For each directory in the upward walk, inspect these filenames:
- `CLAUDE.md`
- `AGENTS.md`
- `README.md`

For each existing candidate, capture:
- path
- matched filename
- whether it is a symlink
- resolved real path
- whether it is a regular file
- file size

Selection rule for the initial version:
- do not reject candidates simply because they are symlinks
- resolve each candidate to its real path
- skip candidates that are not regular files
- deduplicate globally by resolved real path so the same underlying file is only loaded once
- if multiple candidates in the same directory resolve to different real files, load all of them

This matches the intended workflow where `CLAUDE.md` and `AGENTS.md` may be symlink aliases to `README.md`, while still allowing intentionally distinct files to coexist.

## Traversal and precedence

The tool should return files ordered from least specific to most specific:
- highest ancestor first
- nearest ancestor last

This allows broader context to appear first and more local guidance to appear later.

Within a single directory, inspect candidates in this order:
- `CLAUDE.md`
- `AGENTS.md`
- `README.md`

That order preserves the more specialized entrypoint names for metadata and diagnostics, while real-path deduplication prevents duplicate content loads when they point at the same underlying file.

The tool itself does not merge or rewrite instructions. It only returns discovered context and metadata.

## Boundary detection

The tool should support an explicit `stop_at` argument.

Initial boundary behavior:
- if `stop_at` is provided, stop when that directory is reached
- otherwise, stop at the workspace root

A later revision may prefer repository root detection when available.

## Data model changes

No shared schema or persistence changes are required.

The new tool will expose a `ToolSpec` with JSON parameters and return a `ContextPartitionItem` whose `output` is structured text.

Initial tool parameters:
- `path` (required): file or directory path to start from
- `stop_at` (optional): inclusive traversal boundary
- `max_bytes_per_file` (optional): per-file content cap to avoid oversized context

## Protocol changes

### New tool

Add a new tool named `context_loader` to the available tool set.

Suggested tool description:
- load path-local context by walking upward from a file or directory and returning relevant `CLAUDE.md`, `AGENTS.md`, and `README.md` files while deduplicating alias paths that resolve to the same underlying file

### System guidance

Add `system_message_parts` explaining when to use the tool, for example:
- use it before making code changes in a project subtree when local project guidance may matter
- use it before planning or reviewing a feature in a nested workspace directory
- use it before inspecting files in a subtree where local instructions may affect interpretation
- do not use it for trivial requests when path-local context is unlikely to matter

## Redesigned or new functions with pseudocode

### `ContextLoaderTool.call(arguments: str, call_id: str)`

```python
async def call(arguments: str, call_id: str) -> ContextPartitionItem:
    args = json.loads(arguments)
    input_path = Path(args["path"]).resolve()
    stop_at = resolve_stop_at(args.get("stop_at"))
    max_bytes = int(args.get("max_bytes_per_file", DEFAULT_MAX_BYTES))

    start_dir = input_path if input_path.is_dir() else input_path.parent
    walked_dirs = iter_dirs_upward(start_dir, stop_at)

    loaded = []
    skipped = []
    seen_real_paths = set()

    for directory in reversed(walked_dirs):
        for filename in ["CLAUDE.md", "AGENTS.md", "README.md"]:
            candidate = directory / filename
            if not candidate.exists():
                continue

            is_symlink = candidate.is_symlink()
            real_path = candidate.resolve()
            is_file = candidate.is_file()

            if not is_file:
                skipped.append((candidate, "not a regular file"))
                continue
            if real_path in seen_real_paths:
                skipped.append((candidate, f"duplicate real path: {real_path}"))
                continue

            content, truncated = read_with_cap(candidate, max_bytes)
            loaded.append(
                metadata_and_content(
                    matched_path=candidate,
                    matched_name=filename,
                    real_path=real_path,
                    is_symlink=is_symlink,
                    truncated=truncated,
                    content=content,
                )
            )
            seen_real_paths.add(real_path)

    output = render_output(
        query_path=input_path,
        stop_at=stop_at,
        walked_dirs=walked_dirs,
        loaded=loaded,
        skipped=skipped,
    )
    return ContextPartitionItem(call_id=call_id, output=output, type="function_call_output")
```

### Helper: upward directory walk

```python
def iter_dirs_upward(start_dir: Path, stop_at: Path | None) -> list[Path]:
    dirs = []
    current = start_dir
    while True:
        dirs.append(current)
        if stop_at is not None and current == stop_at:
            break
        if current.parent == current:
            break
        current = current.parent
    return dirs
```

### Helper: output rendering

The output should be easy for the model to consume directly. Initial format:
- summary section
- walked directories section
- loaded files section with metadata
- skipped files section with reasons
- file contents section with clear path delimiters

Recommended loaded-file metadata fields:
- matched path
- matched filename
- real path
- is symlink
- byte count read
- truncation flag

## Infrastructure changes

Initial implementation files:
- add `prokaryotes/tools_v1/context_loader.py`
- update `prokaryotes/tools_v1/__init__.py` if tools are re-exported there
- update any tool registration code that instantiates and exposes available tools

Observed likely review targets before implementation:
- `prokaryotes/api_v1/models.py`
- any runtime code that assembles tool lists for the LLM client(s)

## Open questions

1. Should the default boundary be workspace root or repository root?
   - Initial proposal: workspace root.
2. Should output be markdown only, JSON only, or markdown plus an embedded JSON block?
   - Initial proposal: markdown with explicit metadata sections.
3. Should duplicate detection use resolved path only, or stronger file identity metadata such as device and inode when available?
   - Initial proposal: resolved real path is sufficient for the first version.
4. Should very large files be truncated with a marker, or omitted with a warning?
   - Initial proposal: truncate to `max_bytes_per_file` and mark truncation explicitly.
5. Should hidden variants such as `.claude.md` be considered later?
   - Initial proposal: no, not in the initial version.
6. Should callers be able to request only the nearest matching ancestor rather than all ancestors?
   - Initial proposal: no, not in the initial version.

## Initial implementation notes

The first version should optimize for:
- transparency
- predictable traversal
- compatibility with the existing `FunctionToolCallback` contract
- low policy complexity
- faithful support for symlink-alias workflows already present in the repository

The first version should not attempt to:
- interpret or merge instructions semantically
- auto-call itself
- discover arbitrary documentation files beyond the three named candidates
- deduplicate by file content instead of file identity
