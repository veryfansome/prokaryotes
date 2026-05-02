# Open Questions

## Model for sub-calls

Each perspective call is a separate LLM inference. Options:

- **Same model as the harness** — highest quality but adds latency and cost proportional to the number of perspectives.
- **Smaller/faster model (e.g. Haiku)** — lower cost, lower latency, potentially shallower analysis. May be sufficient for most think-call scenarios.
- **Configurable at construction time** — `ThinkTool` accepts a model override, defaulting to a fast model. The harness can inject a stronger model for high-stakes use cases.

The right default is an open question until we can measure quality vs. cost empirically.

## Recursion and cost visibility

When perspectives trigger follow-up think calls, the total number of LLM sub-calls grows. The harness currently tracks `context_pct` usage against the outer model's token window. Inner think calls are invisible to that accounting. Whether this matters depends on how often recursive thinking occurs in practice.
