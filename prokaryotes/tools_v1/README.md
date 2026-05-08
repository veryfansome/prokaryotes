# Tools

`tools_v1/` contains reusable `FunctionToolCallback` implementations. Both LLM clients discover available tools through this protocol.

## `FunctionToolCallback` contract

Implementors must expose:

- `name` — unique tool name used to route LLM tool calls to the correct callback.
- `system_message_parts` — list of guidance strings injected into the system/developer message on every request so the model knows the tool exists and how to use it.
- `tool_spec` — a `ToolSpec` instance with `.to_anthropic_tool_param()` / `.to_openai_function_tool_param()` converters.
- `async call(arguments: str, call_id: str) -> ContextPartitionItem | None` — invoked when the LLM calls the tool. Must return a `ContextPartitionItem` of type `function_call_output`. Returning `None` stops the tool-call loop without feeding any results back — for Anthropic this causes an API error on the next request because Anthropic requires a `tool_result` for every `tool_use`. Always return a real item; use `output="ok"` as a no-op acknowledgment rather than returning `None`.

## Existing tools

- `FileTool` (`file_tool.py`) — reads files, creates new UTF-8 text files, and edits existing files by line range. Read outputs are **live windows** that the harness keeps in sync with on-disk content across turns; mutation outputs are frozen **edit records**. Line-edit actions use `expected_revision` for optimistic concurrency; `create_file` fails if the path already exists and returns a live window of the current file so the model can recover in one step. Requires the active `ContextPartition` and workspace root at construction so writes can refresh prior live windows in-place with the same path-sandbox policy as user-initiated calls; the module also exports `reconcile_tracked_files(context_partition, workspace_root=...)` which each harness calls per turn after `sync_context_partition()` to refresh live windows against current disk state and tombstone any paths that became inaccessible.
- `ThinkTool` (`think.py`) — takes structured `goal`, `context`, and `perspectives` parameters and makes a single focused LLM call with all of them, returning structured analysis the model can act on. Requires an `LLMClient` and model at construction time.
- `ShellCommandTool` (`shell_command.py`) — runs arbitrary shell commands. Output is truncated at 400 lines.
