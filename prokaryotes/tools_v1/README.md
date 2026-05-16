# Tools

`tools_v1/` contains reusable `FunctionToolCallback` implementations. Both LLM clients discover available tools through this protocol.

## `FunctionToolCallback` contract

Implementors must expose:

- `name` — unique tool name used to route LLM tool calls to the correct callback.
- `system_message_parts` — list of guidance strings injected into the system/developer message on every request so the model knows the tool exists and how to use it.
- `tool_spec` — a `ToolSpec` instance with `.to_anthropic_tool_param()` / `.to_openai_function_tool_param()` converters.
- `async call(arguments: str, call_id: str) -> ContextPartitionItem | None` — invoked when the LLM calls the tool. Must return a `ContextPartitionItem` of type `function_call_output`. Returning `None` stops the tool-call loop without feeding any results back — for Anthropic this causes an API error on the next request because Anthropic requires a `tool_result` for every `tool_use`. Always return a real item; use `output="ok"` as a no-op acknowledgment rather than returning `None`.

## Existing tools

- `FileTool` (`file_tool/`) — reads, creates, and edits files by line range. Read outputs are **live windows** that the harness refreshes per turn via the module's `reconcile_tracked_files()` helper. Requires the active `ContextPartition` and workspace root at construction. See [project/features/file_tool/README.md](../../project/features/file_tool/README.md).
- `ThinkTool` (`think.py`) — focused LLM call with structured `goal`, `context`, and `perspectives` parameters. Requires an `LLMClient` and model at construction time. See [project/features/think_tool/README.md](../../project/features/think_tool/README.md).
- `ShellCommandTool` (`shell_command.py`) — runs arbitrary shell commands. Output is truncated at 400 lines.
