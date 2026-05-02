# Persistence and Meta-Learning

## Motivation

The current design treats think calls as ephemeral. The reasoning disappears after the tool-call loop ends. But if the tool is used appropriately — invoked at genuine decision points rather than reflexively — then each think call is a signal that the agent encountered something worth remembering: a non-trivial gap, a constraint it had to reason through, a trade-off it had to weigh. Letting that signal evaporate is a missed opportunity.

Two phases of increasing ambition:

## Phase 1: Persistence and retrospective analysis

### What to store

When `ThinkTool.call()` runs, persist a record containing:

- The structured inputs: `context`, `goal`, `perspectives`
- The assembled output (each perspective's analysis)
- A reference to the `partition_uuid` of the context partition at call time
- A timestamp and session identifier

Elasticsearch is the natural home — the project already uses it to store and search `ContextPartition` records, and full-text search over think call content would be useful for analysis.

### What this enables

**Outcome correlation.** By linking a think call record to the surrounding context partition, we can later ask whether the session that contained it succeeded or failed at its overall task. Over enough sessions, this gives signal on whether think calls at certain kinds of decision points actually helped.

**Usage pattern analysis.** What `goal` strings come up repeatedly? What kinds of `context` descriptions precede a think call? Are there patterns that suggest the tool is being used well (genuine gaps, novel constraints) vs. used reflexively (restating things already in context)?

**Iteration surface.** The current tool description and `system_message_parts` are static. Stored think call records make it possible to audit whether the guidance is working — and to change it based on evidence rather than intuition.

### Relationship to existing infrastructure

- `prokaryotes/search_v1/` — add a `ThinkCallSearcher` or extend `ContextPartitionSearcher` to index think call records alongside partition data.
- `prokaryotes/tools_v1/think.py` — `call()` writes to the store after assembling perspective output, before returning the `ContextPartitionItem`.
- The store write should be fire-and-forget (non-blocking) so it does not add latency to the tool-call loop.

## Phase 2: Thinking frameworks and cross-domain generalization

### The Anthropic result

The [Anthropic engineering post on the think tool](https://www.anthropic.com/engineering/claude-think-tool) reports that pairing it with domain-specific prompting examples drove the largest performance gains — in one case a 54% improvement on τ-bench. The static examples in the system prompt gave the model a template for what a useful think call looked like in that domain.

The hypothesis here: rather than authoring those examples by hand and baking them into the system prompt statically, we can grow them from the think call record store and retrieve them dynamically.

### Thinking frameworks

A *framework* is a reusable pattern for how to approach a class of reasoning problem. Examples:

- "When deciding between implementation options, enumerate the failure modes of each before committing."
- "When a tool output is ambiguous, state explicitly what you expected vs. what you received before deciding how to proceed."
- "When the plan has more than three sequential steps, identify which step is the most fragile and reason about it first."

Frameworks are extracted from past think calls — either by a separate analysis process reviewing stored records and identifying patterns that correlated with good outcomes, or by the agent itself during or after a session.

A framework record would store:
- A natural-language description of the reasoning pattern
- The domain or task type where it was observed (concrete)
- An abstracted version of the pattern (general)
- Evidence: which think call records instantiate it, and what the outcome was
- A retrieval embedding so it can be matched against new `goal` + `context` inputs at call time

### Retrieval at call time

When `ThinkTool.call()` runs, before fanning out perspective sub-calls, retrieve the top-k most relevant frameworks from the store based on the current `goal` and `context`. Inject them into each perspective sub-call prompt as examples — the same mechanism the Anthropic post identified as high-value, but driven by retrieval rather than static authoring.

This means the quality of the think tool improves as the agent accumulates experience, without requiring manual curation of examples.

### Cross-domain generalization

The harder question is whether a reasoning pattern learned in one domain transfers to another. The human analogy is real: someone who has learned to reason carefully about failure modes in one domain (say, distributed systems) often applies the same habit productively in an unrelated domain (say, planning a complex task sequence).

One way to model this:

- Frameworks exist at multiple levels of abstraction. A concrete framework is tied to a specific domain or task type. An abstract framework is a generalization of the pattern stripped of domain-specific content.
- When a concrete framework is applied in a new domain and the outcome is good, that is evidence that the abstract version generalizes. The store can track this provenance.
- Over time, the most abstract and broadly applicable patterns rise to the top of retrieval across many domains. Domain-specific patterns stay relevant within their domain but do not crowd out general ones elsewhere.
- The agent (or a background analysis process) can explicitly perform this abstraction step: "I applied the 'enumerate failure modes first' pattern in a file-system task. I previously used it in a network task. The abstract form seems to be: identify the highest-fragility step in any sequential plan before committing." That abstraction becomes a new framework record.

This is the mechanism by which meta-learning happens: not through gradient descent, but through explicit representation, retrieval, and iterative refinement of reasoning patterns grounded in the agent's own history.

### Role of the agentic loop

The abstraction step and the usage analysis step are both good candidates for a background agentic loop (see [project/ideas/agentic_loop/options.md](../agentic_loop/options.md)) rather than inline work during a session.

**Framework distillation.** A background loop can periodically review recently stored think call records, identify clusters of concrete frameworks with shared structure, and attempt to distill an abstract generalization. This mirrors how humans consolidate experience during idle periods — dwelling on recent events not to replay them but to extract what was transferable. Running this loop at low frequency (e.g. after each N sessions, or on a nightly cadence) means the token cost is amortized against a meaningful batch of evidence rather than spent speculatively on a single observation.

**Think tool effectiveness analysis.** A separate pass in the same loop can look at think call records in aggregate and ask harder evaluative questions: Were goals stated specifically enough to produce focused output? Did any perspective analyses turn out to be redundant? Were there sessions where the tool was called but the output did not visibly change the next action taken? These are signals that the tool description, the `system_message_parts` guidance, or the perspective sub-call prompts need adjustment. The loop can surface findings as annotated records or proposed guidance changes for review, rather than applying them automatically.

The case for spending tokens on this is the same as the case for human reflection: the cost of dwelling is low relative to the value of carrying forward a better reasoning habit into future sessions.

### Open questions for Phase 2

- How do we evaluate whether a retrieved framework actually helped on a given think call? Outcome correlation at the session level is coarse; we may need finer-grained signals.
- How do we prevent framework proliferation — many near-duplicate patterns stored separately? Some form of deduplication or clustering over the embedding space would be needed.
- Should frameworks be scoped per user, per agent instance, or shared globally across all sessions?
- What triggers a background loop run — a fixed schedule, a session count threshold, or an explicit signal from the web harness after a session ends?
