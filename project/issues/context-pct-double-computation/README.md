# Issue: `context_pct` Computed Twice Per LLM Round

## Location

- `prokaryotes/anthropic_v1/__init__.py:85-95` — computed in `stream_turn`
- `prokaryotes/openai_v1/__init__.py:98-104` — computed in `handle_response_stream_event`
- `prokaryotes/anthropic_v1/web_harness.py:93-97` — recomputed in `on_usage`
- `prokaryotes/openai_v1/web_harness.py:93-97` — recomputed in `on_usage`

## Problem

After each LLM round, `context_pct` is calculated in two separate places using the same token count:

**In the LLM client** (for the stream event):
```python
# anthropic_v1/__init__.py
context_window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
context_pct = int(total_input / context_window * 100)
if stream_ndjson:
    yield json.dumps({"context_pct": context_pct}) + "\n"
```

**In the harness's `on_usage` callback** (for the compaction threshold):
```python
# anthropic_v1/web_harness.py
def on_usage(input_tokens: int, output_tokens: int) -> None:
    context_window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
    context_pct = int(input_tokens / context_window * 100)
    if context_pct >= COMPACTION_TOKEN_THRESHOLD_PCT:
        pending_compaction[0] = True
```

Both expressions divide the same token count by the same context window and multiply by 100. The LLM client already has the value when it calls `on_usage` — it passes `total_input` as the first argument — but the harness recomputes it instead of receiving it directly.

This also forces the harness to import `MODEL_CONTEXT_WINDOWS` and `DEFAULT_CONTEXT_WINDOW` solely for this arithmetic, creating an import dependency that would go away if the client passed the percentage instead.

## Proposed Fix

Change the `on_usage` callback signature to accept `context_pct` as a third parameter. The LLM client computes it once and passes it:

```python
# LLM client (both providers) — call on_usage with the already-computed percentage
if on_usage is not None:
    on_usage(total_input, response.usage.output_tokens, context_pct)
```

The harness callback becomes:

```python
# anthropic_v1/web_harness.py and openai_v1/web_harness.py
def on_usage(input_tokens: int, output_tokens: int, context_pct: int) -> None:
    if context_pct >= COMPACTION_TOKEN_THRESHOLD_PCT:
        pending_compaction[0] = True
```

The `Callable[[int, int], None]` type on the `on_usage` parameter in both LLM clients changes to `Callable[[int, int, int], None]`. The harnesses can then remove their imports of `MODEL_CONTEXT_WINDOWS` and `DEFAULT_CONTEXT_WINDOW` (assuming those constants aren't needed elsewhere in those files).

### Tradeoff

This couples the callback signature to the concept of context percentage, which is a compaction-specific concern that the LLM client doesn't otherwise know about. An alternative is to keep the current signature and have the harness compute `context_pct` from `input_tokens`, accepting the duplication. Given that the LLM client already contains the context-window lookup table (`MODEL_CONTEXT_WINDOWS`) and performs the same computation for the stream event, passing the result to `on_usage` is the more natural choice.

---

## Second Opinion

**Verdict: Overstated. The proposed fix is the wrong direction. There is a real but different bug here.**

The reviewer agreed the double computation is factually accurate, then dismissed it: the arithmetic is one integer division and one multiplication — the computational cost is literally immeasurable. Calling it an "inefficiency" overstates it. It is cosmetic duplication only.

**The proposed fix worsens separation of concerns.** Embedding `context_pct` into the `on_usage` callback signature permanently couples a transport-layer interface to a compaction policy concept. The LLM client's context-window lookup table is already a concern that arguably doesn't belong there — making `context_pct` part of the callback contract makes this coupling explicit and permanent. The tradeoff section above acknowledges the coupling but incorrectly concludes in favour of the fix.

**The fix also has wider blast radius than stated.** Four additional call sites would need updating beyond the two harnesses: `anthropic_v1/script_harness.py:31`, `openai_v1/script_harness.py:31`, `eval_v1/harness.py:86`, and both LLM client `__init__.py` type annotations. The issue omits these entirely.

**The real bug is a provider token-count asymmetry the issue missed.** The Anthropic client assembles `total_input` by summing `input_tokens + cache_read_input_tokens + cache_creation_input_tokens` (lines 85-89) before calling `on_usage`. The OpenAI client passes only `usage.input_tokens` with no cache-token accumulation (line 101). Whether OpenAI returns cache tokens in a separate field is worth investigating. This asymmetry could cause the compaction threshold to fire at different effective usage levels across providers on cached requests — an actual correctness concern.

**Recommendation:** Close this issue. Open a separate issue for the Anthropic/OpenAI token-count asymmetry in the `on_usage` call.
