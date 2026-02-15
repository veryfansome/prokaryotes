# Persistent Agentic Loop Options

## Problem statement

The current web harness is reactive at the outermost level. The agent becomes active when a user calls `POST /chat`, runs through one bounded response/tool loop, and then stops. There is no always-on internal controller that can wake up between user prompts, inspect the world, reconsider its state, and take action on its own.

If we want the agent to have meaningful internal goals, we likely need some form of persistent loop.

## Current baseline in this repo

Relevant facts from the current implementation:

- The web app is a long-lived FastAPI process with lifecycle hooks in `prokaryotes/web_v1/__init__.py`:
  - `WebBase.init()`
  - `WebBase.lifespan()`
  - `WebBase.on_start()`
  - `WebBase.on_stop()`
- The current chat flow is request-scoped:
  - `openai_v1/web_harness.py` and `anthropic_v1/web_harness.py` both create a `StreamingResponse` around `stream_and_finalize(...)`
  - `stream_and_finalize(...)` runs the LLM response loop and then only does finalize/compaction housekeeping
- The repo already has durable backing services:
  - Postgres
  - Redis
  - Elasticsearch
  - Neo4j
- There is already a non-web headless execution primitive:
  - `prokaryotes/openai_v1/script_harness.py`
  - `prokaryotes/anthropic_v1/script_harness.py`
- `docker-compose.yml` already models multiple long-lived services, so adding a worker process would fit the existing deployment shape.

Important limitation:

- `ScriptHarness.run()` calls `os.chdir(cwd)`, which is process-global. The eval harness explicitly runs tasks sequentially for this reason. A persistent worker built directly on `ScriptHarness` should therefore start single-threaded unless that behavior is refactored.

Important UI constraint:

- The current UI is built around request/response chat streaming plus compaction-status polling. There is no general push channel for unsolicited background agent activity.

## Option 1: Request-coupled pseudo-persistence

### Idea

Keep the system reactive, but recompute the agent's internal agenda every time the user sends a message.

### What it buys

- Minimal implementation cost
- No new long-lived worker
- Easy to layer onto the current web harness

### What it does not buy

- No true between-turn agency
- No autonomous wake-ups
- No action without a user prompt

### Best use

Good if the actual goal is "the agent appears self-regulating when spoken to" rather than "the agent genuinely continues to think and act between prompts."

## Option 2: In-process web poller

### Idea

Start a forever-loop task from `WebBase.on_start()` inside the web process. That loop wakes periodically, inspects persistent state, and performs agent work.

### What it buys

- Fastest path to a real persistent loop
- No new service to deploy
- Naturally reuses existing Postgres/Redis/Search connections

### Risks

- Lifecycle is tied to the web server
- Harder to reason about if multiple web replicas exist later
- Crashes or deploy restarts interrupt agent work
- Background loop and user-serving path compete for the same process

### Best use

Good for early experimentation, especially if the app is single-instance and we want to validate loop semantics before introducing more infrastructure.

## Option 3: Post-response bounded continuation

### Idea

After a user prompt finishes, schedule extra bounded work via `background_and_forget(...)`. This lets the agent keep working for a while after the HTTP response returns.

### What it buys

- Very small change from the current architecture
- Useful for short follow-through chains
- Preserves a strong causal connection to the user turn that triggered the work

### Limits

- Not a general always-on runtime
- Work only begins because a user spoke first
- Fragile across restarts
- Hard to treat as a true persistent internal loop

### Best use

Good for "continue for a bit after the turn ends," not for durable internal goal pursuit.

## Option 4: Separate worker service

### Idea

Add a dedicated long-lived process such as `scripts/agent_worker.py`, and run it as a separate service alongside `web` in `docker-compose.yml`.

### What it buys

- Cleanest separation of concerns
- True home for persistent internal goals
- Better operational story than hiding a loop inside the web server
- Easier to evolve toward scheduling, retries, locks, and supervision

### Costs

- More moving parts
- Need a persistence contract for pending work and agent outputs
- Need some way to surface background activity in the UI

### Best use

Best serious path if we mean "persistent agentic loop" literally.

## Option 5: Schedule-driven worker

### Idea

Instead of a hot continuous loop, run a worker on a fixed cadence. It can wake every few seconds, every minute, or on some coarser schedule.

### What it buys

- Simpler than an event-rich always-hot controller
- Often enough for many internal goals
- Easy to reason about and test

### Limits

- Less responsive than event-driven wakeups
- Can feel sluggish if low latency matters

### Best use

Good first real implementation if internal goals mostly involve checking stale state, re-evaluating commitments, or resuming deferred follow-through.

## Option 6: Event-driven queue worker

### Idea

Use Redis or Postgres as a lightweight event queue. The web app and other subsystems append wakeup signals such as:

- user turn finished
- tool failed
- assumption invalidated
- observation became stale
- deadline crossed

The worker consumes those signals and decides what to do.

### What it buys

- More responsive than periodic polling
- Better fit for internal goals that should wake on specific changes
- Avoids constant scanning when nothing interesting happened

### Costs

- More coordination logic
- Need idempotency and deduplication
- Slightly more operational complexity than a simple timer loop

### Best use

Best when we want the agent to feel continuously alive without implementing a tight busy loop.

## Comparison

| Option | True between-turn agency | Complexity | Operational robustness | Fit for this repo |
|---|---|---|---|---|
| Request-coupled pseudo-persistence | No | Low | High | Very easy |
| In-process web poller | Yes | Low-medium | Medium-low | Easy |
| Post-response bounded continuation | Weakly | Low | Low-medium | Easy |
| Separate worker service | Yes | Medium | High | Strong |
| Schedule-driven worker | Yes | Medium | High | Strong |
| Event-driven queue worker | Yes | Medium-high | High | Strong |

## Recommended path

If the goal is genuine internal agent goals rather than merely richer request-time behavior, the best path is:

1. Start with a separate worker service.
2. Make it schedule-driven first.
3. Add event-driven wakeups once the state model is clear.

This sequence keeps the architecture honest without overbuilding too early.

## Recommended v1 shape

### Runtime

- Add a dedicated worker entrypoint, e.g. `scripts/agent_worker.py`
- Run it as a new `agent-worker` service in `docker-compose.yml`
- Use a simple single-threaded loop at first

### State

- Store canonical internal-goal state in Postgres
- Use Redis for locks, wakeup signals, or light queueing
- Optionally use Elasticsearch only for projection/search, not as canonical

### Execution

- Reuse `ScriptHarness` initially for headless action
- Treat it as single-task-at-a-time unless `os.chdir(...)` is removed from the execution path

### UI / observability

- Do not rely on `/chat` as the only surface
- Add a separate persisted activity/inbox model for background agent outputs
- Later, add polling or push for that activity feed

## Open design questions

- Should the worker run one global loop, or one loop per scope/user/project?
- What kinds of actions are allowed without a fresh user prompt?
- What is the canonical persistence model for internal goals and pending work?
- Should internal goals compile into the same prompt digest machinery used for chat turns, or a separate worker-facing agenda?
- How should background agent activity appear in the UI without feeling like fake chat messages?

## Bottom line

Within this project, there are several ways to simulate persistence, but only two really match the idea of a continuing agent:

- an in-process web poller
- a dedicated worker service

For experimentation, the web poller is the quickest path. For a real internal goal architecture, a dedicated worker service is the better foundation.
