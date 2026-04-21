# Tools

`tools_v1/` contains reusable `FunctionToolCallback` implementations. Both LLM clients discover available tools through this protocol.

## `FunctionToolCallback` contract

Implementors must expose:

- `name` — unique tool name used to route LLM tool calls to the correct callback.
- `system_message_parts` — list of guidance strings injected into the system/developer message on every request so the model knows the tool exists and how to use it.
- `tool_spec` — a `ToolSpec` instance with `.to_anthropic_tool_param()` / `.to_openai_function_tool_param()` converters.
- `async call(arguments: str, call_id: str) -> ContextPartitionItem | None` — invoked when the LLM calls the tool. Must return a `ContextPartitionItem` of type `function_call_output`. Returning `None` stops the tool-call loop without feeding any results back — for Anthropic this causes an API error on the next request because Anthropic requires a `tool_result` for every `tool_use`. Always return a real item; use `output="ok"` as a no-op acknowledgment (as `ThinkTool` does) rather than returning `None`.

## Existing tools

- `ThinkTool` (`think.py`) — gives the model a private reasoning scratchpad between tool calls. Returns `output="ok"` as a no-op acknowledgment.
- `ShellCommandTool` (`shell_command.py`) — runs arbitrary shell commands. Output is truncated at 400 lines.
