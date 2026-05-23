# Think Tool

## Implementation

`prokaryotes/tools_v1/think.py` implements `ThinkTool` with four structured parameters:

- **`context`** (string, required) — relevant prior tool outputs, constraints, partial results, and facts that should inform the next decision.
- **`goal`** (string, required) — the specific gap or decision that triggered this think call (not a restatement of the overall task).
- **`perspectives`** (list of strings, required) — named lenses (e.g. `"implementation options"`, `"order of operations"`, `"risks"`) to analyze `context` against `goal`. Empty list = single synthesis.
- **`paths`** (list of strings, required) — workspace-relative or absolute file paths whose active `WorkingFileWindow`s should be injected into the think subprompt. Empty list = no file context.

Surfacing `context` and `goal` as explicit slots forces the model to commit to what it knows and what it's missing before reasoning forward, rather than letting both stay implicit.

### Active reasoning call

`call()` makes a single focused LLM call with `context`, `goal`, all perspectives, and (when `paths` is non-empty and a `working_file_provider` is wired) the matching live `WorkingFileWindow`s assembled into a structured prompt. The full conversation history is excluded — the call operates only on what the model explicitly provides. This keeps the reasoning focused on the local gap rather than letting the model re-derive things that are already settled.

Working-file injection is gated: only windows whose path resolves under `workspace_root` AND whose `status == "live"` are included. If the outer model names a path with no active window, the inner call proceeds without that file's content. The wrapping block uses a per-call nonce delimiter (`<active-working-files-{nonce}>…</active-working-files-{nonce}>`) for injection resistance — the think subprompt is built outside `project_for_llm`, so it doesn't benefit from the projection seam's closing-tag escape discipline.

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

Recursive think calls are bounded by the outer `max_tool_call_rounds` limit enforced by the LLM clients.

## Relationship to the codebase

- `prokaryotes/tools_v1/think.py` — the implementation. `ThinkTool` requires an `LLMClient` and model at construction; `reasoning_effort` defaults to `low` or the `THINK_TOOL_REASONING_EFFORT` env var. Optional `working_file_provider` (typically `lambda: conversation.working_file_windows`) and `workspace_root` enable the `paths` parameter's working-file injection.
- `prokaryotes/api_v1/models.py` — `FunctionToolCallback` contract: `call()` returns a `TurnItem` with `type="function_call_output"` carrying the assembled analysis.
- Both LLM clients (`anthropic_v1/`, `openai_v1/`) route the think call to `ThinkTool.call()` and feed the returned `TurnItem` back into the tool-call loop.
