# Think Tool: Structured, Active Reasoning

## Current state

`prokaryotes/tools_v1/think.py` implements `ThinkTool` as a no-op scratchpad. The model calls it with a single free-form `thought` string, receives `output="ok"`, and continues. The tool description and `system_message_parts` guidance shape *when* to use it but place no constraints on *what* goes into the thought.

In practice this means thoughts can be verbose, repetitive, or loosely connected to the immediate decision the model needs to make — spending tokens on context that is already visible in the conversation rather than on focused reasoning about a specific gap.

## Problem

Two related issues with the current design:

1. **Unconstrained content.** A single string accepts anything. There is no signal to the model about what a *useful* thought looks like relative to the specific moment in a task.

2. **No output.** The tool is a pure sink. The model injects a thought and gets back nothing. This means the thinking itself cannot be acted on by any other subsystem, and there is no way to evaluate whether the reasoning actually helped.

## Proposed design

### Parameter schema

Replace the single `thought` string with three structured parameters:

- **`context`** (string, required) — all data points the model considers relevant at this moment: prior tool outputs, constraints, partial results, and any facts that should inform the next decision. Forces the model to state what it actually knows before reasoning forward.

- **`goal`** (string, required) — what the model needs before it can move on. Not a restatement of the overall task; specifically the gap or decision that triggered this think call. Grounds the reasoning in a concrete local objective rather than a general reflection.

- **`perspectives`** (list of strings, optional) — lenses through which to analyze the `context` relative to the `goal`. Each perspective is a named angle such as `"implementation options"`, `"order of operations"`, or `"risks"`. When supplied, each perspective is analyzed in a separate focused LLM call run concurrently.

### Active reasoning call

Instead of returning `output="ok"` immediately, `call()` makes one or more focused LLM calls:

- Each call receives only `context` + `goal` + a single perspective label as its prompt. The full conversation history is excluded. This keeps the reasoning focused on the local gap rather than letting the model re-derive things that are already settled.

- When `perspectives` is empty or omitted, a single synthesis call runs with just `context` + `goal`, producing a general focused analysis rather than a multi-angle breakdown.

- All perspective calls run concurrently. Their results are assembled into a structured output that the calling model receives as the tool result, with each perspective clearly labeled.

### Example output shape

```
[implementation options]
Option A: iterate over existing results and filter in-place. Risk: mutates shared state.
Option B: collect into a new list. Cleaner but allocates. Preferred given the small result set.

[risks]
The upstream tool result may be empty if the search returned no matches. The plan does not currently handle that case — should add a guard before proceeding.
```

The calling model reads this output and decides its next step. If any perspective raises an open question it cannot resolve from context, it may invoke the think tool again with a narrower goal targeting that question specifically.

## Iterative use

The think tool is not limited to one call per task. The design explicitly allows the model to call it again when a perspective surfaces uncertainty that warrants deeper reasoning. Each subsequent call should carry a more specific `goal` derived from the open question raised in the prior output.

This creates a lightweight recursive structure: think → act or think again → act. The number of recursive think calls is implicitly bounded by the overall `max_tool_call_rounds` limit already enforced by the LLM clients.

## Relationship to the current codebase

- `prokaryotes/tools_v1/think.py` — the implementation target. `ThinkTool.call()` gains the focused LLM call logic. `tool_spec` gains the new parameter schema. `system_message_parts` gains guidance on how to populate `context`, `goal`, and `perspectives`.
- `prokaryotes/api_v1/models.py` — `FunctionToolCallback` contract is unchanged; `call()` still returns a `ContextPartitionItem` with `type="function_call_output"`. The output field carries the assembled perspective results rather than `"ok"`.
- Both LLM clients (`anthropic_v1/`, `openai_v1/`) are unchanged. They route the think call to `ThinkTool.call()` and feed the returned `ContextPartitionItem` back into the tool-call loop as they do now.

## What this does not change

- The calling model still decides what to do with the output. The think tool surfaces analysis; it does not issue instructions or constrain next steps.
- The `max_tool_call_rounds` limit remains the only bound on total think usage within a session.

## Further directions

See [persistence_and_meta_learning.md](persistence_and_meta_learning.md) for ideas on storing think call records for retrospective analysis, and on growing reusable thinking frameworks from accumulated experience that can be retrieved and injected at call time.
