# File Tool Sweep And Recovery Worker

This directory is for design work around **post-compaction repair**.

The immediate motivation is the current compaction path for tracked file context:

1. persist a child partition in Elasticsearch as `compaction_state="pending"`
2. swing the Redis head to that child with CAS
3. promote the child to `compaction_state="committed"`
4. mark the parent partition compacted with the same `compaction_attempt_uuid`

That order avoids publishing a Redis head that Elasticsearch cannot load, but it still leaves two classes of unfinished state that we may want to repair later:

- a stale `pending` child when the CAS never commits
- a live child whose parent was never marked compacted because a post-CAS ES write failed

This matters for file-tool state because the compacted child can carry the lifted tail and tracked-file annotations that later rebuilds need when the Redis cache expires.

## Current contract

The codebase now records two pieces of compaction metadata on context-partition docs:

- `compaction_state`: `pending` or `committed`
- `compaction_attempt_uuid`: a per-attempt correlation id shared by the child and, on success, the parent update

Normal discovery paths already treat `pending` docs as non-canonical:

- exact `get_partition(partition_uuid)` loads still work
- search and tail-hash lookup filter to `committed` or legacy docs with no state field

That means a future worker can safely inspect `pending` docs without them showing up in normal search results.

## Important definition: what counts as an orphan

A `pending` child is **not** automatically an orphan.

It is only safe to treat a child as abandoned when we can show all of the following:

- it is older than a grace period
- it is not the active Redis head for the conversation
- it is not an ancestor of the active Redis head
- the parent was not compacted through the same `compaction_attempt_uuid`

That third rule is important. If the conversation already advanced past the child, Redis will point at a descendant, not the child itself. In that case the child is still part of the live chain and must be repaired, not swept.

## Recommended split: recovery first, sweeper second

These should be treated as two related but different jobs.

### Recovery worker

The recovery worker is the correctness mechanism. Its job is to make Elasticsearch eventually reflect a compaction that already became live.

The worker should repair these cases:

- `pending` child is on the active ancestry chain
  - promote the child to `committed`
  - if the parent is still uncompacted, replay the parent compacted update with the same `compaction_attempt_uuid`
- `committed` child is on the active ancestry chain but the parent is still uncompacted
  - replay only the parent compacted update
- parent already carries the same `compaction_attempt_uuid` but the child is still `pending`
  - promote the child to `committed`

### Orphan sweeper

The sweeper is a storage-hygiene mechanism. Its job is to retire stale `pending` docs that never became part of the active chain.

The sweeper should only touch docs that the recovery logic has already ruled out as live. For the first version, it is reasonable to make this conservative:

- scan only `compaction_state="pending"` docs older than a grace window
- skip anything whose liveness cannot be proven either way
- prefer logging or a soft state transition before hard deletion if we want an extra safety phase

## What the worker needs to know

The current schema is almost enough for a first pass.

Existing signals we can already use:

- `partition_uuid`
- `conversation_uuid`
- `parent_partition_uuid`
- `compaction_state`
- `compaction_attempt_uuid`
- `dt_created` / `dt_modified`
- parent `is_compacted`, `summary`, `boundary_*`, and `compaction_attempt_uuid`
- the current Redis head for `context_partition:{conversation_uuid}`

The key runtime capability the worker needs is:

- given a conversation head UUID, walk the ES ancestor chain by `parent_partition_uuid`

That ancestry walk must use exact partition loads, not search, because search intentionally hides `pending` docs.

## Decision procedure

For a candidate `pending` child:

1. Load the current Redis head for the conversation.
2. If there is no Redis head, do not sweep the doc.
3. Starting from the Redis head UUID, walk the ancestor chain in ES.
4. If the candidate child appears anywhere in that chain, it is live.
5. If the candidate child is live:
   - promote it to `committed` if still `pending`
   - replay the parent compacted update if needed
6. If the candidate child is not live:
   - load the parent doc
   - if the parent already carries the same `compaction_attempt_uuid`, treat the child as recoverable, not orphaned
   - otherwise, once the grace period has elapsed, mark it eligible for sweeping

For a candidate `committed` child with an uncompacted parent:

1. Load the Redis head and active ancestry chain.
2. If the child is on the active chain, replay the parent compacted update.
3. If the child is not on the active chain, leave it alone.

That last rule avoids rewriting old branches just because they happen to be committed.

## Why Redis-head equality is not enough

A naive sweeper might ask only whether Redis points directly at the child UUID. That is not sufficient.

Example:

- compaction CAS succeeds and Redis points at child `c1`
- later requests extend the conversation and finalize `c2`, `c3`, ...
- the current Redis head is now `c3`

At that point `c1` still belongs to the active branch, even though it is no longer the current head. The worker has to reason over the whole active ancestry chain, not a single UUID comparison.

## Extra metadata that could help later

The current fields are enough to start, but these would make operations easier if we ever want better observability:

- `compaction_recovered_at`
- `compaction_recovery_error`
- `compaction_abandoned_at`

None of those are required for the first version.

## Suggested implementation shape

The safest rollout is:

1. Add a repair helper that operates on one conversation head at a time.
2. Call that helper from a background worker or a request-time recovery hook.
3. Only after recovery is working, add a periodic pending-doc scan for sweep candidates.

That keeps correctness and cleanup separate:

- recovery prevents rebuild regressions and preserves tracked file state
- sweeping keeps the index tidy once recovery can distinguish live from stale attempts

## Test cases we should require before enabling cleanup

- `pending` child is the Redis head and gets promoted to `committed`
- `pending` child is an ancestor of the Redis head and still gets promoted
- `committed` child with uncompacted parent causes the parent update to replay
- stale `pending` sibling created by a lost CAS race is not on the active chain and becomes sweepable after the grace period
- worker does not delete or rewrite anything when the Redis head is missing
- multiple `pending` children under the same parent do not confuse attempt matching

## Non-goals for the first version

- a cross-store transaction between Redis and Elasticsearch
- aggressive deletion of ambiguous docs
- cleaning up old committed branches

The first version should be biased toward **false negatives** in sweeping, not false positives. Leaving an extra `pending` doc around is much cheaper than deleting a live compacted branch that contains tracked file state.
