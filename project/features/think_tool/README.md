# Think Tool: Structured, Active Reasoning

## Implementation

`prokaryotes/tools_v1/think.py` implements `ThinkTool` with three structured parameters:

- **`context`** (string, required) — all data points the model considers relevant at this moment: prior tool outputs, constraints, partial results, and any facts that should inform the next decision. Forces the model to state what it actually knows before reasoning forward.

- **`goal`** (string, required) — what the model needs before it can move on. Not a restatement of the overall task; specifically the gap or decision that triggered this think call. Grounds the reasoning in a concrete local objective rather than a general reflection.

- **`perspectives`** (list of strings, required) — lenses through which to analyze the `context` relative to the `goal`. Each perspective is a named angle such as `"implementation options"`, `"order of operations"`, or `"risks"`. Pass an empty list when a single synthesis is sufficient.

### Active reasoning call

`call()` makes a single focused LLM call with `context`, `goal`, and all perspectives assembled into a structured prompt. The full conversation history is excluded — the call operates only on what the model explicitly provides. This keeps the reasoning focused on the local gap rather than letting the model re-derive things that are already settled.

When `perspectives` is non-empty, the system prompt instructs the LLM to address each perspective in a dedicated labeled section in the order listed. All perspectives are handled in the same call so they can inform one another rather than being analyzed in isolation.

The output is returned as the tool result; the calling model reads it and decides its next step.

### Example output shape

```
[implementation options]
Option A: iterate over existing results and filter in-place. Risk: mutates shared state.
Option B: collect into a new list. Cleaner but allocates. Preferred given the small result set.

[risks]
The upstream tool result may be empty if the search returned no matches. The plan does not currently handle that case — should add a guard before proceeding.
```

If any perspective raises an open question it cannot resolve from context, it may invoke the think tool again with a narrower goal targeting that question specifically.

## Iterative use

The think tool is not limited to one call per task. The design explicitly allows the model to call it again when a perspective surfaces uncertainty that warrants deeper reasoning. Each subsequent call should carry a more specific `goal` derived from the open question raised in the prior output.

This creates a lightweight recursive structure: think → act or think again → act. The number of recursive think calls is implicitly bounded by the overall `max_tool_call_rounds` limit already enforced by the LLM clients.

## Relationship to the codebase

- `prokaryotes/tools_v1/think.py` — the implementation. `ThinkTool` requires an `LLMClient` and model at construction; `reasoning_effort` defaults to `low` or the `THINK_TOOL_REASONING_EFFORT` env var.
- `prokaryotes/api_v1/models.py` — `FunctionToolCallback` contract is unchanged; `call()` returns a `ContextPartitionItem` with `type="function_call_output"` carrying the assembled analysis.
- Both LLM clients (`anthropic_v1/`, `openai_v1/`) are unchanged. They route the think call to `ThinkTool.call()` and feed the returned `ContextPartitionItem` back into the tool-call loop.
