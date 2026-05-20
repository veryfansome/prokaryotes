# Conversation Compaction

Compaction is a lifecycle operation on the [conversation model](../conversation/README.md): it lets a conversation run indefinitely past the provider's context-window limit by periodically summarizing the oldest history into a compact, LLM-generated briefing. This doc covers the compaction-specific machinery; the `Conversation` primitives, reconciliation, the wire protocol, and branching are documented in [conversation/README.md](../conversation/README.md).

---

## Core Concepts

### Snapshot chains

Before compaction a conversation is a single flat `Conversation` snapshot. Compaction introduces a **linear chain** within the snapshot DAG:

```
[P0: compacted] → [P1: compacted] → [P2: active]
```

A compacted snapshot is sealed (`is_compacted=True`) and carries an LLM-generated `summary`. The active snapshot is the working set and the only one cached in Redis at `conversation:{conversation_uuid}`. (The DAG also has *branch* edges from edit/regenerate — see [conversation/README.md](../conversation/README.md#branches-and-snapshots).)

When the model is called, context is assembled as:

```
[core instructions] + [summary block from compacted ancestors] + [projected active snapshot]
```

Summaries enter via the system (Anthropic) / developer (OpenAI) `instruction` parameter as a trailing `# Compacted conversation summary` background-memory section — never as conversation turns, so they aren't conflated with real dialogue.

### Ancestor summaries

Each `Conversation` carries `ancestor_summaries` — ordered oldest-first, immutable, riding with the snapshot through Redis. The ES chain walk is paid once per branch reconstruction; subsequent requests take the Redis fast path.

### Recency tail

On compaction, the most recent K user/assistant messages stay verbatim in the new active snapshot (`COMPACTION_RECENCY_TAIL`, default 6). `raw_message_start_index` tracks the boundary — the count of non-deleted messages folded into the compacted ancestor chain. The child's `messages` holds only the recency tail plus anything appended while the summary was in flight; the compacted prefix is reconstructed from the parent chain when needed.

### Boundary hashing

A summary is only valid if the branch that produced it is provably identical to the current branch. Each compacted snapshot stores a `boundary_hash` — a SHA-256 over the `(author_id, content)` sequence of every non-deleted message the summary covers. On chain reconstruction the server recomputes the hash over the corresponding prefix of incoming and compares; a mismatch means the branches diverged before the boundary and the summary is not used. A secondary `tail_hash` (last N user-message contents) is a lookup key for tail-based discovery.

### Lifted items and the file-tool anchor

When compaction commits, the child's `lifted_turn_items` carry forward file-tool live windows (and their paired `function_call`s) for any path still active in the recency tail. `lifted_anchor_source_id` is the recency-tail bot message that touched a relevant path — projection emits the lifted items immediately before that bot's tool round. Invariant: `anchor=None iff lifted_turn_items==[]`. The summary input has live-window bodies stripped (`strip_live_window_bodies`) so current file contents don't fossilize into the summary.

---

## Compaction-specific storage

The `conversations` ES index (defined in [conversation/README.md](../conversation/README.md#elasticsearch)) carries these compaction fields:

| Field | Purpose |
|---|---|
| `compaction_state` | `committed` or `pending` (child written before the CAS). Search and tail-hash discovery ignore `pending`. |
| `compaction_attempt_uuid` | Correlates the child + parent writes of one attempt. |
| `is_compacted` | Set on the parent only after the child swap commits. |
| `summary` | LLM-generated summary text; full-text indexed. |
| `boundary_hash` / `tail_hash` | Branch-validation hash and tail lookup key. |
| `boundary_message_count` / `boundary_user_count` | Counts at the compaction boundary. |
| `ancestor_summaries` / `lifted_turn_items_json` / `lifted_anchor_source_id` | Mirrors of the in-memory fields. |

A compacted snapshot keeps its `messages_json` so the [assistant-message guardrail](../conversation/README.md#assistant-message-guardrail) can still read compacted ancestors' per-message identity.

---

## Compaction Flow

### Trigger

`WebHarness._dispatch_turn` registers an `on_usage` callback. After each LLM round it computes `context_pct = input_tokens / context_window * 100` and sets the request-local `pending_compaction` flag once it meets `COMPACTION_TOKEN_THRESHOLD_PCT` (default 80%).

### Sequencing in `stream_and_finalize`

After the final assistant message commits:

1. `finalize_turn` appends the bot `ConversationMessage`, persists `Conversation` + `TurnExecution`, and refreshes the assistant index.
2. Emit `bot_message`.
3. If `pending_compaction` is set, `SET NX EX` `compaction_lock:{conversation_uuid}`. On success: emit `compaction_pending`, deep-copy the snapshot, and fire `_compact_conversation` as a background task.

### `_compact_conversation` — `prokaryotes/context_v1/compaction.py`

1. **`_prepare_compaction`** — split `(pre_tail, recency_tail)`, load `TurnExecution`s for both windows, build the `_LiftPlan`. Pure read step.
2. Return early if the recency tail is empty.
3. **Summarize** via `compact_fn` — `_summarize_and_compact` projects the pre-tail with stripped live-window bodies, appends the summarization prompt, and calls `llm_client.complete(...)`. An empty summary aborts.
4. **Compute boundary metadata** — `boundary_hash`, `tail_hash`, the counts.
5. **`_cas_swap_child`** — under Redis `WATCH / MULTI / EXEC` on `conversation:{conversation_uuid}`: abort if the active `snapshot_uuid`, `ancestor_summaries`, or `raw_message_start_index` changed; build the child (`messages = recency_tail + post_snapshot_messages`, `ancestor_summaries + [summary]`, lift state from prep); persist it as `compaction_state="pending"` so Redis never points at a snapshot ES doesn't know; execute the swap, retry on `WatchError`.
6. **Promote the child to `committed`** via `update_conversation`. This write uses `refresh="wait_for"` so a post-compaction cold-Redis `_rebuild_from_chain` can immediately find the new active child via `find_latest_active_child`. The wait runs on this background task — no interactive latency.
7. **Mark the parent `is_compacted`** with the summary + boundary metadata.
8. **Write `compaction_status:{pending_snapshot_uuid}`** to Redis — the child's `snapshot_uuid` as the relabel target, or an empty-string sentinel on abort/exception.
9. **Release the lock** (always, via `finally`).

The three ES writes are each wrapped in `_retry_compaction_search_write` (retry-with-backoff). The CAS swap guarantees messages sent during the summary call are carried forward as `post_snapshot_messages`.

### Compaction-status polling

`compaction_pending` is the last stream event when a background compaction started; the UI shows a non-blocking indicator. The client polls `GET /compaction-status?conversation_uuid=...&pending_snapshot_uuid=...` every 5 seconds. On `{done: true, snapshot_uuid}` it runs `relabelSnapshotUuid` to move the affected `messageTree` nodes onto the child snapshot, then clears the indicator.

The indicator is **branch-scoped**: each pending compaction has its own poll loop keyed on the scheduling `snapshot_uuid`, so concurrent sibling-branch compactions don't interfere and switching branches mid-compaction never strands a relabel. Polling is the **only** clearing path — a transient poll failure keeps the loop alive, and there is no side-channel clear on subsequent stream handshakes (unsafe across branches).

---

## Compaction and branching

Each branch is independently compactable — the CAS is keyed on `snapshot_uuid`. How edit/retry interacts with the compaction boundary:

- **Edit/regenerate within the recency tail** → Case A divergence: a new branch snapshot that inherits `ancestor_summaries` and recomputes lift state.
- **Retry before the recency tail** → `_split_compacted_prefix` fails, the syncer routes to Case B: a fresh root branch with no ancestor summaries and `raw_message_start_index=0`.

See [conversation/README.md](../conversation/README.md#branches-and-snapshots) for the full divergence semantics.

---

## Key Design Decisions

**Child snapshot persisted as `pending` before the CAS.** Redis never points at a snapshot ES doesn't know about; discovery filters out `pending` docs until they commit.

**`refresh="wait_for"` is scoped to the child-committed write only.** Conversation writes are otherwise eventually consistent for search — forcing a refresh on every turn's `put_conversation` would add ~1–2s of latency to every `/chat` turn. The find-after-write race only matters for `find_latest_active_child` on a post-compaction cold-Redis rebuild, so the one write that gates that search waits; it runs on the background compaction task at no interactive cost.

**Compaction-status relabel is the only indicator-clearing path.** An earlier design cleared on a subsequent stream handshake carrying a different snapshot UUID — unsafe once branches exist (a handshake on branch B would clear branch A's indicator).

**Compaction lock prevents duplicate concurrent compactions.** Only the first request to acquire `compaction_lock:{conversation_uuid}` compacts; the CAS swap carries the loser's messages forward regardless.

---

## Known Limitations and Deferred Work

- **Deep rehydration.** Retries before the recency tail build a fresh branch with no ancestor summaries. The original message lists survive in ES (`messages_json` is kept) but aren't used to restore full fidelity.
- **Script and eval harnesses don't compact.** Long non-interactive sessions that approach the context limit fail.
- **Pending-child cleanup.** A `pending` child whose CAS never commits lingers in the index; discovery filters it out, but a sweeper is deferred.

---

## Relevant Code Files

| File | Role |
|---|---|
| `prokaryotes/context_v1/compaction.py` | `ConversationCompactor`: `_compact_conversation`, `_cas_swap_child`, `_prepare_compaction`, `_write_compaction_status`; `_compute_lift_plan`, `_recency_tail_messages`, `_retry_compaction_search_write`. |
| `prokaryotes/context_v1/conversation_sync.py` | `_split_compacted_prefix`, `_rebuild_from_chain`, `find_latest_active_child` use — the chain-rebuild and compacted-prefix handling. |
| `prokaryotes/harness_v1/web.py` | `on_usage` / `pending_compaction`, `_summarize_and_compact`. |
| `prokaryotes/harness_v1/base.py` | `stream_and_finalize` compaction sequencing. |
| `prokaryotes/web_v1/compaction.py` | `CompactionStatusHandler`: the `/compaction-status` endpoint. |
| `prokaryotes/tools_v1/file_tool/live_windows.py` | Live-window strip/lift helpers used by the lift plan. |
| `prokaryotes/utils_v1/llm_utils.py` | `COMPACTION_TOKEN_THRESHOLD_PCT`, `COMPACTION_RECENCY_TAIL`, `COMPACTION_LOCK_TTL_SECONDS`. |
| `tests/unit_tests/test_compaction_*.py` | Lift plan, CAS swap, chain rebuild, status handler, summarization input. |
| `tests/integration_tests/tier_b/test_compaction_flow.py` | End-to-end compaction against real ES + Redis. |
