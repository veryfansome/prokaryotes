# Issue: Ancestor Summary Injection Is Asymmetric Across Providers

> **Status: Resolved** by the `ContextPartition` → `Conversation` migration. `WebHarness._build_instruction_parts` now appends `conversation.ancestor_summary_block()` for both providers, and the providers receive the unified `instruction` string via `llm_client.stream_turn(items=..., instruction=..., ...)`. There is no provider-specific summary injection on the data-model side anymore.

## Location (historical)

- `prokaryotes/harness_v1/web.py` — `WebHarness._build_instruction_parts()` manually appended summaries only for `impl="openai"`
- `prokaryotes/api_v1/models.py` — the old `ContextPartition.to_anthropic_messages()` folded in `ancestor_summaries`

## Problem (historical)

Ancestor summaries need to reach the model as part of the system/developer context on every request. The consolidated web harness handles the two providers differently:

**Anthropic** — `to_anthropic_messages()` automatically folds `partition.ancestor_summaries` into the system string:

```python
def to_anthropic_messages(self):
    system_parts: list[str] = list(self.ancestor_summaries)
    ...
```

The Anthropic web-harness path does not need to touch `ancestor_summaries` in its instruction parts.

**OpenAI** — `to_openai_input()` does not include `ancestor_summaries`. The OpenAI branch must remember to append them manually:

```python
# harness_v1/web.py
if self.impl == "openai":
    summary_block = context_partition.ancestor_summary_block()
    if summary_block:
        parts.append("")
        parts.append(summary_block)
```

This asymmetry creates a latent footgun: implementing a third provider requires the author to know (from reading the OpenAI branch or the compaction docs) that ancestor summaries must be manually injected. Nothing in the `ContextPartition` API signals this requirement — `to_openai_input()` silently omits the summaries with no indication that the caller is responsible for them.

If a new provider path omits this step, conversations with compacted history will silently lose context with no error.

## Proposed Fix

Add a method to `ContextPartition` that provides the assembled preamble for providers that need to build their own context header, making the contract explicit:

```python
# api_v1/models.py

def ancestor_summary_preamble(self) -> str | None:
    """Returns the ancestor summaries joined for injection into a developer/system message,
    or None if there are no summaries. Providers that build their own context header
    (e.g. OpenAI) must include this; Anthropic providers use to_anthropic_messages()
    which handles it automatically."""
    if not self.ancestor_summaries:
        return None
    return "\n\n".join(self.ancestor_summaries)
```

The OpenAI branch then calls this explicitly:

```python
developer_message_parts = []
preamble = context_partition.ancestor_summary_preamble()
if preamble:
    developer_message_parts.append(preamble)
developer_message_parts.append("# Tool usage")
...
```

This makes the injection visible and self-documenting. The docstring in `ancestor_summary_preamble` signals to future provider implementors that they are responsible for calling it.

An alternative is to add a note to `to_openai_input()`'s docstring that it does not include ancestor summaries and callers must inject them via the developer message. That's lower effort but less discoverable.

---

## Second Opinion

**Verdict: Valid problem, but the proposed fix does not solve the footgun.**

The reviewer confirmed the asymmetry: `to_anthropic_messages()` folds in `ancestor_summaries` automatically, while the OpenAI branch in `harness_v1/web.py` manually appends them. This is real and correctly described.

**`ancestor_summary_preamble()` does not solve the footgun.** The method is advisory and opt-in. A future provider author who does not read its docstring will still omit the call and silently lose compacted context. The proposed fix converts an implicit contract (read the OpenAI harness to know you must inject summaries) into a slightly more explicit one (read the docstring of an optional utility method). That is a discoverability improvement, not a solution.

**The only structural solutions** would be to make injection impossible to omit — either by having `to_openai_input()` prepend a synthetic dict unconditionally (mechanically possible, since `ancestor_summaries` can be prepended as a `{"role": "developer", ...}` dict), or by requiring callers to explicitly dispose of the summaries through the API. The issue does not explore these.

**The asymmetry is partly structurally forced.** Anthropic's API takes a top-level `system` string, which is why `to_anthropic_messages()` naturally assembles the full system content. OpenAI's Responses API takes `input: list[dict]` with a first-class `developer` message item that `WebHarness` constructs locally. Some provider-specific context-header code in the harness is unavoidable given these API shapes. This weakens the framing of the asymmetry as a bug rather than a natural consequence of provider differences.

**Recommendation:** Reframe this as a discoverability problem with a partial fix. Add `ancestor_summary_preamble()` and a docstring on `to_openai_input()` stating it does not include ancestor summaries. Do not claim this solves the footgun.
