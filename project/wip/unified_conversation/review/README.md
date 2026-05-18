# Unified Conversation — Overlay Review

Review of [`../overlay/`](../overlay/) against [`../README.md`](../README.md), synthesized from two independent passes (Claude + Codex). Findings are ordered by severity. Each entry names the offending file and line(s), explains the failure mode, and proposes a concrete fix grounded in the design doc.

The overlay's own verification commands all pass — `uv run ruff check` is clean, 66/66 overlay Python unit tests pass, 12/12 vitest JS tests pass. The issues below are integration-level: they live in the seams between the unit-tested helpers, and they materialize only on flows the overlay's tests don't exercise (post-compaction round-trip, regenerate, branch creation, startup, CLI). The doc's Tier-B `test_unified_web_flow.py` would surface most of them; it's listed in the design doc but not implemented in the overlay.

If the docker-compose data stores are up (elasticsearch, postgres, redis), verifying fixes integration-side is a one-command path: bring up the stack, run the overlay's Tier B suite once we add it. Pair the Tier B coverage with focused fake-Redis / fake-Search unit tests for the prefix splitter and branch-policy decisions — the splitter's correctness is too central to gate only on integration tests.

---

## 1. Blocker — Post-compaction append corrupts the snapshot

**Where:** [`overlay/prokaryotes/context_v1/conversation_sync.py:180`](../overlay/prokaryotes/context_v1/conversation_sync.py) (`_apply_result` append branch), reached via the normal flow in [`sync_conversation`](../overlay/prokaryotes/context_v1/conversation_sync.py).

**Failure mode.** After a compaction completes and the client runs `relabelSnapshotUuid`, every `messageTree` node retains its content and gets its `snapshot_uuid` updated to the child snapshot's id. The client's `buildRequestMessages` walks the active path from root to leaf — so the next POST sends `messages=[m1..m7, m8]` (every message the client knows about), even though the stored child snapshot's `messages` only contains the recency tail `[m5,m6,m7]` (m1..m4 are summarized into `ancestor_summaries`).

Walking the syncer:

1. Tier 1 Redis hit: cached snapshot has `messages=[m5,m6,m7]`, `_conversation_can_follow_client` returns True, the cached snapshot is returned without any check against `raw_message_start_index` or `ancestor_summaries`.
2. `_detect_unacknowledged_bot_messages` returns `[]` (no trailing bot after m7).
3. `reconcile`: `shared_prefix=[m5,m6,m7]`, `m1..m4,m8` are emitted as `append` ops (none of them are in `stored_by_id`). `classify` sees `kinds={"append"}`, `all_stored_in_incoming=True`, `all_contents_match=True` → `"append"`.
4. `_apply_result` blindly appends m1..m4 and m8 to `stored.messages` (line 183).

Result: `stored.messages=[m5,m6,m7,m1,m2,m3,m4,m8]`. After sort in `project_for_llm` the LLM sees m1..m4 expanded again, and the same content remains in `ancestor_summary_block` via the instruction message — content is duplicated, every turn, forever, growing across every subsequent compaction.

The doc anticipates this — `_rebuild_from_chain` already implements the boundary-hash validation against compacted ancestors and produces a fresh `Conversation` with the right `raw_message_start_index` — but only when Tier 1 and Tier 2 both miss. In the happy post-compaction path Tier 1 hits, and the chain rebuild never runs.

**Proposed fix.** Add a *compacted-prefix split* step **before `_detect_unacknowledged_bot_messages`** (and therefore before reconcile). The split has to run first because compacted-prefix entries can otherwise show up as "new content" in the resync detector's `has_new_incoming` check and distort the resync decision.

> **Why not `boundary_hash`?** The parent's `boundary_hash` covers the parent's *full* visible history at compaction time — pre-tail + raw tail. After a later Case A branch that modifies the raw tail, the incoming history won't match `boundary_hash` even though it should. The hash is only valid as an exact-match check against the parent at the moment of compaction.
>
> The correct check is "does incoming's compacted-away prefix match what the server has on record?" — using the reconstructed prefix content directly, not the parent boundary hash.

```python
# In sync_conversation, immediately after _load_stored returns `stored`:
if stored.raw_message_start_index > 0:
    prefix_match, raw_suffix = await self._split_compacted_prefix(stored, partial)
    if not prefix_match:
        # Case B: the user is editing inside the compacted prefix. Discard `stored`
        # and produce a fresh Conversation rooted at the incoming list, with no
        # inherited ancestor_summaries / lifted state. Source-id assignment for
        # bare incoming entries still happens here — see note below.
        return await self._build_case_b_result(conversation_uuid, bot_author_id, partial)
    # Normal post-compaction: drop the prefix the server already has summarized,
    # reconcile only the raw-window suffix against stored.messages.
    partial = raw_suffix

# Resync detection now operates on the raw-window suffix only.
unacknowledged = _detect_unacknowledged_bot_messages(stored, partial)
if unacknowledged:
    return SyncResult(...)

# Assign source_ids to any raw-suffix entries that arrived without one (newly-
# typed user messages). _to_normalized would otherwise raise on source_id=None.
assignments = self._assign_source_ids(partial, stored)
normalized = [_to_normalized(m) for m in partial]

# Reconcile sees the same raw-window suffix — any divergence is Case A.
result = reconcile(stored, normalized)
```

Where `_split_compacted_prefix(stored, partial)`:

1. Walks the parent chain to reconstruct the `expected_prefix` — the first `stored.raw_message_start_index` non-deleted `ConversationMessage`s from the *global* history. This is the same walk `_boundary_messages_for_conversation` already does; lift that helper into shared code (the compactor and the syncer both need it).
2. Takes the first `stored.raw_message_start_index` non-deleted entries from `partial` (call it `incoming_prefix`).
3. Compares `(source_id, author_id, content)` triples between `expected_prefix` and `incoming_prefix`. If lengths differ or any triple differs → return `(False, partial)`.
4. Otherwise return `(True, partial[stored.raw_message_start_index:])`.

The `source_id` is included in the comparison deliberately — accepting different source-ids with the same content would weaken the source-ID-as-identity invariant the rest of the system relies on. A client that has lost track of its prior source_ids will fail the split and fall to Case B, which produces a fresh `Conversation` rooted at incoming — exactly the right recovery.

This makes Tier 1's Redis hit safe: the cached snapshot is correct, the syncer drops the client-side prefix because the server already has its summary, and only the raw-window suffix flows through both the resync detector and reconcile.

Case B classification stays bound to "the compacted-prefix content didn't match" — *not* to a `divergence_point_index` comparison. After the split, any divergence reconcile sees is by definition within the raw window — Case A.

**Case B helper contract.** `_build_case_b_result` must still:

1. Run `_assign_source_ids(partial, _empty_conversation)` so any bare incoming entries get monotonic source_ids assigned against an empty conversation (not against the discarded `stored`). The `SourceIdAssignment` entries it returns are what the handshake's `source_id_assignments` carries back to the client; skipping this step strands the client without `source_id` for its fresh message.
2. Map those assignments back to the original `client_index` values from `_partially_normalize`, so the handshake's positions still line up with the request's `messages` array.
3. Build the fresh `Conversation` from the now-assigned `partial`, return a `SyncResult` with `is_new_branch=True` and the populated `source_id_assignments`.

The pseudocode above shows an early return for clarity, but the real implementation should structure these steps as a `_build_case_b_result` helper that takes `partial`, runs assignment, persists, and produces the full `SyncResult`. Don't let an early-return short-circuit skip assignment.

**Tests to add.**
- `test_compacted_prefix_split.py::test_matching_prefix_strips_cleanly` — fake search client returns a parent doc whose first `raw_message_start_index` non-deleted messages match incoming's first N entries on `(source_id, author_id, content)`; split returns `(True, raw_suffix)`.
- `test_compacted_prefix_split.py::test_content_mismatch_routes_to_case_b` — same shape but one `(source_id, author_id, content)` triple differs in `content`; split returns `(False, partial)`.
- `test_compacted_prefix_split.py::test_source_id_mismatch_routes_to_case_b` — content matches but `source_id` differs (client lost track of prior IDs); split returns `(False, partial)` so we don't silently re-anchor.
- `test_compacted_prefix_split.py::test_length_mismatch_routes_to_case_b` — incoming has fewer than `raw_message_start_index` non-deleted entries; split returns `(False, partial)`.
- `test_compacted_prefix_split.py::test_walks_multi_compaction_chain` — three-deep compaction chain; split correctly reconstructs prefix across all ancestors.
- `test_conversation_syncer.py::test_split_runs_before_resync_detection` — stored has trailing un-acknowledged bot beyond the raw window; client sends compacted prefix + raw-window match + new user message. With the split running first, the resync check sees only the raw-window suffix and correctly flags the trailing bot; without it, the prefix entries pollute `has_new_incoming` and the resync detection misfires.
- `test_conversation_syncer.py::test_post_compaction_append_uses_raw_suffix` — integration of split + reconcile + apply: stored has `messages=[m5,m6,m7]`, `raw_message_start_index=3`; client sends `messages=[m1..m7, m8]` with matching prefix; after sync, `stored.messages == [m5,m6,m7,m8]`, `ancestor_summaries` unchanged.
- `test_conversation_syncer.py::test_case_b_helper_assigns_source_ids` — client sends `messages=[m1', m2', new_user]` (m1', m2' have source_ids that don't match stored compacted prefix; new_user has no source_id); split returns `(False, partial)`; Case B helper assigns a `source_id` to `new_user`, returned in `SyncResult.source_id_assignments` with the correct `client_index`.
- Tier B `test_unified_web_flow.py::test_compaction_relabel_and_continue` — full round-trip: new conversation → repeated turns until compaction → relabel → next turn projects the bot's full history exactly once.

---

## 2. Blocker — Missing web assistant-message guardrails

**Where:** [`overlay/prokaryotes/harness_v1/web.py:91`](../overlay/prokaryotes/harness_v1/web.py) (POST `/chat` body passed straight to `sync_conversation`) and [`overlay/prokaryotes/context_v1/conversation_sync.py:601`](../overlay/prokaryotes/context_v1/conversation_sync.py) (`_partially_normalize` maps `role="assistant"` → `bot_author_id` unconditionally).

**Failure mode.** The design doc requires: "A future POST whose `messages` list contains an assistant `source_id` that doesn't appear in *any* ConversationMessage for this `conversation_uuid` (across the entire snapshot DAG, including compacted ancestors), or a known assistant `source_id` whose content differs from what storage has, is corruption — not a legitimate edit — and the route handler rejects with a 4xx."

The overlay has no such check. `_partially_normalize` accepts any `role="assistant"` entry and stamps it with `bot_author_id`; reconcile then either treats it as a new bot message (Append) or as an edit of an existing bot's content. A malicious or buggy client can fabricate bot history or rewrite past assistant turns. The Case B retry-before-recency-tail flow (which legitimately echoes compacted ancestors' assistant source_ids) explicitly requires DAG-scoped recognition — also missing.

**Proposed fix.** Introduce a DAG-scoped assistant index, checked in `post_chat` before invoking `sync_conversation`. Two implementation options listed in the doc; a Redis-cached `{source_id: content_hash}` mapping is the cheaper hot path:

```python
# In WebHarness.post_chat, before sync_conversation:
assistant_index = await self._load_assistant_index(incoming.conversation_uuid)
for msg in incoming.messages:
    if msg.role != "assistant":
        continue
    if msg.source_id is None:
        raise HTTPException(400, "Assistant messages must carry server-assigned source_id")
    known = assistant_index.get(msg.source_id)
    if known is None:
        raise HTTPException(400, f"Unknown assistant source_id: {msg.source_id}")
    if known.content_hash != sha256(msg.content):
        raise HTTPException(400, f"Assistant content mismatch for {msg.source_id}")
```

`_load_assistant_index(conversation_uuid)` reads `assistant_index:{conversation_uuid}` from Redis (a serialized `{source_id: content_hash}` dict). The index is populated on first conversation load by walking the snapshot DAG (`search_client.find_conversation_messages_for_conversation` — new ES query, `terms` filter on the conversations index returning `messages_json`) and updated in `finalize_turn` whenever a new bot message commits. Compacted ancestors are reachable because the design preserves their `messages_json` after `is_compacted=true`.

**Tests to add.**
- `test_web_wire_protocol.py::test_unknown_assistant_source_id_rejected` — POST with a fabricated assistant `source_id` → 400.
- `test_web_wire_protocol.py::test_assistant_content_tampering_rejected` — POST with a known assistant `source_id` but different content → 400.
- `test_web_wire_protocol.py::test_case_b_retry_with_compacted_assistant_accepted` — POST whose assistant `source_id` comes from a compacted ancestor → accepted (DAG-scoped lookup hits the compacted snapshot's retained `messages_json`).

---

## 3. High — Case B branches inherit `ancestor_summaries` / lifted state incorrectly

**Where:** [`overlay/prokaryotes/context_v1/conversation_sync.py:208`](../overlay/prokaryotes/context_v1/conversation_sync.py) `_apply_divergence`.

**Failure mode.** When reconcile classifies as `divergence`, `_apply_divergence` is invoked. It unconditionally inherits the parent's `ancestor_summaries` and `raw_message_start_index`. The doc explicitly distinguishes Case A (divergence within the parent's raw window — the common edit/regenerate path; inherit compacted-prefix state) from Case B (divergence before the parent's raw window — retry-before-recency-tail; produce a fresh `Conversation` with no inherited compacted-prefix state, `raw_message_start_index=0`).

The current implementation has no case discrimination — every divergence is treated as Case A. The docstring on line 220 claims "the Case B path materializes in `_rebuild_from_chain`'s fresh-Conversation fallback, never here," but the only way to reach that fallback is a Tier 1 + Tier 2 cache miss; in the happy path (`stored` loaded successfully) we never get there.

**Proposed fix.** Don't try to detect Case B inside `_apply_divergence` — by the time reconcile runs we've already lost the global index needed to distinguish "edited the first raw-window message" from "edited a compacted-away message." Both can produce `divergence_point_index == 0`.

Instead, fold Case B detection into the compacted-prefix split step from Issue 1. After the split runs:

- **Split succeeded** (prefix matched, or `raw_message_start_index == 0`): reconcile sees only the raw-window suffix. Any divergence it reports is by construction Case A. `_apply_divergence` inherits compacted state and recomputes lifted state (per Issue 8).
- **Split failed** (prefix content mismatch): the user is editing inside the compacted prefix — that's Case B. The syncer skips reconcile entirely and produces a fresh `Conversation` from the full incoming list, with `ancestor_summaries=[]`, `lifted_turn_items=[]`, `lifted_anchor_source_id=None`, `raw_message_start_index=0`.

With that wiring, `_apply_divergence` itself simplifies — it never needs to special-case Case B because Case B never reaches it. The existing Case A logic still needs the lifted-state fix from Issue 8.

**Tests to add.**
- `test_conversation_syncer.py::test_case_b_edit_in_compacted_prefix_produces_fresh_conversation` — stored is a compacted-child snapshot with `raw_message_start_index > 0`; client edits a message in the compacted prefix (split fails); result is a fresh `Conversation` with empty `ancestor_summaries`, empty `lifted_turn_items`, `raw_message_start_index=0`.
- `test_conversation_syncer.py::test_case_a_edit_first_raw_message_inherits_compacted_state` — same stored snapshot; client edits the first raw-window message (`divergence_point_index == 0` *after the split*); result inherits `ancestor_summaries`, recomputes lifted state, and is a Case A branch — *not* misclassified as Case B.
- `test_conversation_syncer.py::test_case_a_edit_inside_raw_window_inherits_compacted_state` — same stored snapshot; client edits a mid-raw-window message; same Case A expectations.

---

## 4. High — Web regenerate / prefix retry tombstones in place instead of branching

**Where:** [`overlay/prokaryotes/context_v1/conversation_sync.py:193`](../overlay/prokaryotes/context_v1/conversation_sync.py) `_apply_result` delete branch; classified in [`reconcile.py`](../overlay/prokaryotes/conversation_v1/reconcile.py) `_classify`.

**Failure mode.** A regenerate-from-U2 request sends `messages=[..., U2]` (omitting the trailing bot reply B2 the client is regenerating against). Reconcile produces `shared_prefix=[..., U2]`, one delete op for B2, `last_shared=U2.source_id`. The trailing-only-delete classifier returns `"delete"`. `_apply_result` then sets `B2.deleted=True` *on the live snapshot* (line 198). The parent branch is destructively mutated — there's no longer a snapshot in ES with B2 visible, breaking the snapshot DAG branch contract ("the original snapshot stays in ES intact").

Slack `message_deleted` legitimately wants in-place tombstone behavior. Web regenerate does not. The current classification is surface-agnostic; the apply policy is shared with no override.

**Proposed fix.** Promote per-surface apply policy to the syncer subclass. The default (web) syncer should treat trailing deletes the same as divergence — branch on a fresh snapshot:

```python
async def _apply_result(self, *, stored, result, ...):
    if result.classification == "match":
        return stored
    if result.classification == "append":
        ...
    if result.classification in {"edit", "delete", "divergence"}:
        return await self._apply_divergence(stored=stored, result=result, ...)
    ...
```

The Slack syncer overrides `_apply_result` (or just the `edit`/`delete` cases) to mutate in place — which is exactly what the doc says: "Slack reconciles a thread in place, web edit/regenerate spawns a new branch snapshot." The "edit" classification carries the same problem (an edit applies in-place today even for web).

**Tests to add.**
- `test_conversation_syncer.py::test_web_regenerate_creates_branch_snapshot` — stored has `[U1, B1, U2, B2]` on snapshot s-1; POST with `messages=[U1, B1, U2]`; result is a new `Conversation` with `parent_snapshot_uuid=s-1`, `messages=[U1, B1, U2]`, and `search_client.get_conversation("s-1")` still shows B2 non-deleted.
- `test_conversation_syncer.py::test_web_edit_creates_branch_snapshot` — analogous for `"edit"`.
- `test_slack_syncer.py::test_slack_message_changed_in_place` — verifies Slack subclass preserves in-place semantics for the same wire shape.

---

## 5. High — Script / eval harness fall-through imports removed `ContextPartition` types

**Where:** [`overlay/prokaryotes/harness_v1/__init__.py:5`](../overlay/prokaryotes/harness_v1/__init__.py) (fall-through into `__path__`), [`prokaryotes/harness_v1/script.py:6`](../../../../prokaryotes/harness_v1/script.py) (upstream, still imports `ContextPartition`, `ContextPartitionItem`).

**Failure mode.** Confirmed by `import prokaryotes.harness_v1.script` under the overlay's `conftest.py` bootstrap:

```
ImportError: cannot import name 'ContextPartition' from 'prokaryotes.api_v1.models'
(/Users/fzhu/.../overlay/prokaryotes/api_v1/models.py)
```

The overlay's `harness_v1/__init__.py` walks the parent chain so unchanged sibling modules (`eval.py`, `script.py`) fall through to upstream. But upstream's `script.py` imports `ContextPartition` and `ContextPartitionItem` — which the overlay's `api_v1/models.py` deliberately removed. `eval.py` then imports from `script.py` and inherits the breakage.

The overlay is effectively *web-only*. `scripts/cli.py` and `scripts/eval.py` (the CLI and eval entry points called out in `CLAUDE.md`) cannot start under the overlay. The migration plan calls this work out as step 4 ("Update `HarnessBase`. Per-harness `stream_and_finalize` accepts the new types") but the overlay doesn't carry the migrated `script.py` / `eval.py`.

**Proposed fix.** Port `harness_v1/script.py` and `harness_v1/eval.py` into the overlay against the new model. The migration is more than mechanical — `EvalHarness` reaches into the partition shape in several places:

- `ScriptHarness.run` (the actual method name — not `run_single_task`) builds a single-turn `Conversation` (no `IncomingConversation` wire trip — `ScriptHarness` synthesizes locally), calls `project_for_llm`, drives `llm_client.stream_turn` exactly the way `WebHarness._dispatch_turn` does, and discards the result without persistence (script runs are non-interactive).
- `EvalHarness.run_task` currently does four things off the partition: reads `partition.items` to compute `tool_call_count` and `think_count`, calls `count_turns(partition.items)` for `turn_count`, and writes the partition itself to `context_partition.json` for post-hoc inspection. Each needs a replacement:
  - `tool_call_count` / `think_count` — derive from the `TurnExecution` list collected during the run (function-call items live there now, not on the conversation).
  - `count_turns` — rewrite against the projected item stream (function-call boundaries are still detectable; assistant messages still separate turns).
  - `context_partition.json` artifact — replace with a new shape that captures both the `Conversation` and its `TurnExecution`s. Probably `conversation.json` + `turn_executions.json`, or a single combined `eval_run.json` with both. Pick one and update the eval tooling that consumes these artifacts.
- `WORKSPACE_ROOT = Path("/tmp/prokaryotes_eval")` and the per-task isolation logic carry forward unchanged.

**Tests to add.**
- `test_script_harness.py::test_run_completes` — synthesize a one-shot task, verify the harness produces a final assistant text and clean teardown. Uses fake LLM client.
- `test_eval_harness.py::test_count_turns_matches_legacy` — fixture a `ProjectedItem` list with known turn boundaries, verify the count matches the old partition-item-based count for equivalent shapes.
- `test_eval_harness.py::test_run_task_writes_artifact_with_conversation_and_turns` — fake LLM run; assert the new artifact captures both `Conversation` and `TurnExecution` content.

---

## 6. High — ES bootstrap (`search_init.py`) still creates the old `context-partitions` index

**Where:** [`scripts/search_init.py:7`](../../../../scripts/search_init.py) — upstream, not migrated; not present in the overlay.

**Failure mode.** The overlay introduces two new ES indices (`conversations` and `turn-executions`) via [`overlay/prokaryotes/search_v1/conversations.py:34`](../overlay/prokaryotes/search_v1/conversations.py). But `scripts/search_init.py` — the bootstrap entry point invoked by `docker compose` on startup — still creates only `context-partitions` and `topics`. Anyone running `docker compose up --build` against the overlay gets a working web process but no ES schema for the new model, so the first `put_conversation` either fails or auto-creates a non-strict mapping that drifts from the design.

**Proposed fix.** Add `overlay/scripts/search_init.py` that mirrors upstream but swaps the schema map:

```python
from prokaryotes.search_v1.conversations import (
    CONVERSATIONS_INDEX,
    TURN_EXECUTIONS_INDEX,
    conversation_mappings,
    turn_execution_mappings,
)
from prokaryotes.search_v1.topics import topic_mappings

schemas = {
    CONVERSATIONS_INDEX: conversation_mappings,
    TURN_EXECUTIONS_INDEX: turn_execution_mappings,
    "topics": topic_mappings,
}
```

The rest of `sync_mappings` (idempotent create-or-update, stop-word analyzer, replica count) stays identical. The legacy `context-partitions` index is *not* created — clean break, per the doc's Out-of-Scope-for-v1 stance.

**Tests to add.** This is a script test — start the docker stack, run `python -m scripts.search_init`, then assert `es.indices.exists("conversations")` and `es.indices.exists("turn-executions")`. Probably lives in Tier B integration tests, not unit tests.

---

## 7. Medium — Bot tombstone re-keying is absent

**Where:** [`overlay/prokaryotes/context_v1/conversation_sync.py:193`](../overlay/prokaryotes/context_v1/conversation_sync.py) `_apply_result` delete branch.

**Failure mode.** The doc spends a paragraph on this: when a bot `ConversationMessage` that owns a `TurnExecution` is tombstoned (Slack admin cleanup, retention reaping), the syncer must re-key `TurnExecution.bot_message_source_id` to the next non-tombstoned bot in the same consecutive run; `Conversation.lifted_anchor_source_id` follows the same re-key rule when its target is tombstoned; if every bot in the run is tombstoned, the `TurnExecution` is orphaned (sweep candidate) and the lift anchor falls to `None`.

The overlay's delete branch is six lines that only flip `msg.deleted = True`. None of the re-keying or anchor-following happens. Once a bot message gets deleted, the projection silently drops its tool-call history (the lookup `historical_turns[msg.source_id]` no longer matches anything reachable via the run-membership rule), and any lifted file-tool window anchored at that bot vanishes.

This is also closely tied to Issue 4 — the right place to fix this is the same surface-specific apply path. Slack's in-place delete is where re-keying happens; web's branch-on-delete sidesteps it but still needs the lift-anchor rule when a regenerate causes the prior anchor bot to no longer exist on the new branch.

**Proposed fix.** Add a helper `_rekey_for_tombstone(stored, deleted_source_id)` that:

1. Finds the consecutive bot run that `deleted_source_id` belongs to (walking forward and backward from the deleted index in `sorted_messages` order, stopping at non-bot messages — tombstoned bots stay in the run for *membership* but are skipped for *selection*).
2. If `deleted_source_id` owns a `TurnExecution`: pick the next non-tombstoned bot in the run as the new owner. Because `turn-executions` uses `bot_message_source_id` as the ES `_id` ([`overlay/prokaryotes/search_v1/conversations.py:308`](../overlay/prokaryotes/search_v1/conversations.py)), the re-key has to *move* the document, not patch a field. Add a `search_client.rekey_turn_execution(old_id, new_id)` method that round-trips the doc (or atomic-deletes+puts under the new id; ES has no native id rename). If the run has no non-tombstoned bots, just `delete_turn_execution(old_id)` — the turn is orphaned.
3. If `stored.lifted_anchor_source_id == deleted_source_id`: apply the same selection rule against the run; if no replacement, set to `None` (and clear `lifted_turn_items` to preserve the invariant `anchor=None iff lifted==[]`).

Wire it into the Slack subclass's `_apply_result` delete branch and into the (new) `message_deleted` handler.

**Tests to add.** The doc's testing plan names five sub-cases of `test_reconcile.py`; we add a sixth (already-deleted-middle-bot) to cover the run-membership-vs-selection distinction:

- `test_tombstone_rekey_single_bot` — bot run of one; tombstone the owner; `TurnExecution` becomes orphaned, anchor falls to `None`.
- `test_tombstone_rekey_chain` — bot run of three; tombstone the first; `TurnExecution`'s ES document moves to the second bot's source_id (fake search client asserts both the old-id delete and the new-id put).
- `test_tombstone_full_run_orphans_turn_execution` — bot run of two; tombstone both; old `TurnExecution` is deleted, no replacement put.
- `test_tombstone_user_message_no_rekey` — tombstoning a user message doesn't trigger any re-keying or ES touch.
- `test_tombstone_bot_without_turn_execution_no_op` — tombstoning a non-owner bot (continuation post) doesn't touch the `TurnExecution` or anchor.
- `test_tombstone_skips_already_deleted_bots_in_run` — bot run of three with middle already tombstoned; tombstoning the first re-keys past the tombstoned middle to the third (membership includes the tombstoned middle but selection skips it).

---

## 8. Medium — Case A branch creation drops lifted file-tool state

**Where:** [`overlay/prokaryotes/context_v1/conversation_sync.py:233`](../overlay/prokaryotes/context_v1/conversation_sync.py) `_apply_divergence` lifted-state recompute; [`overlay/prokaryotes/context_v1/conversation_sync.py:458`](../overlay/prokaryotes/context_v1/conversation_sync.py) `_active_paths_for_messages` stub; [`overlay/prokaryotes/context_v1/conversation_sync.py:628`](../overlay/prokaryotes/context_v1/conversation_sync.py) `_recompute_lifted_anchor` stub.

**Failure mode.** The doc's Case A pseudocode loads historical turns for the shared prefix and recomputes `child_active_paths` against them, then filters parent `lifted_turn_items` by that path set and picks an anchor. The overlay's `_apply_divergence` calls these helpers with empty turn-maps (line 233 passes `{}` for `historical_turns`), and the helpers themselves are stubs that return `set()` / `None`. Every web branch creation therefore produces `lifted_turn_items=[]` and `lifted_anchor_source_id=None`, regardless of what the parent had.

The visible consequence: any tracked file the user was looking at on the parent branch disappears from the LLM's view on the regenerated branch — no warning, no re-read prompt; the file just becomes invisible until the model proactively re-reads. Subtle but real, especially in tool-heavy conversations.

**Proposed fix.** Implement the helpers per the doc's pseudocode. Issue 3 already routes Case B into the prefix-split step, so `_apply_divergence` only ever sees Case A — no `divergence_point_index` check is needed here:

```python
async def _apply_divergence(self, *, stored, result, normalized, bot_author_id, conversation_uuid):
    # Build shared_messages in shared_prefix_source_ids order — not by set membership
    # over stored.messages — so the branch snapshot preserves source-id ordering even
    # if stored.messages's underlying order is ever imperfect.
    stored_by_id = {m.source_id: m for m in stored.messages}
    shared_messages = [stored_by_id[sid] for sid in result.shared_prefix_source_ids if sid in stored_by_id]
    new_tail = [...]

    # Load historical turns for the shared prefix so lift state can be recomputed.
    shared_bot_ids = [m.source_id for m in shared_messages
                      if not m.deleted and m.author_id == bot_author_id]
    child_raw_window_turns = await self.search_client.get_turn_executions(
        stored.conversation_uuid, shared_bot_ids,
    )
    child_active_paths = _active_paths_in_turns(child_raw_window_turns)

    lifted = _filter_lifted_pairs_by_paths(stored.lifted_turn_items, child_active_paths)
    anchor = _first_bot_with_file_activity(shared_messages, child_raw_window_turns, bot_author_id) if lifted else None
    return Conversation(
        ..., lifted_turn_items=lifted, lifted_anchor_source_id=anchor,
    )
```

`_active_paths_in_turns` and `_first_bot_with_file_activity` are direct lifts of the compactor's `_compute_lift_plan` internals — refactor those helpers out so both the compactor and the divergence path share them.

**Tests to add.**
- `test_conversation_syncer.py::test_case_a_branch_preserves_lifted_for_active_paths` — parent has `lifted_turn_items=[(call,output)]` for `/a` and shared prefix contains a bot turn that touches `/a`; new branch retains the lifted pair; anchor is the touching bot.
- `test_conversation_syncer.py::test_case_a_branch_drops_lifted_when_path_not_active` — parent has lifted for `/a`; shared prefix has no `/a` activity; new branch has `lifted_turn_items=[]`, `lifted_anchor_source_id=None`.
- `test_conversation_syncer.py::test_case_a_branch_preserves_multiple_windows_per_path` — parent has two lifted pairs for `/a`; both carry forward.

---

## Suggested order of operations

1. **Land the fake-Redis / fake-Search test fixtures and Tier-B harness in parallel.** Unit-test coverage for the compacted-prefix splitter and branch-policy decisions wants fakes, not real infra; the Tier-B `test_unified_web_flow.py` (compaction-relabel-continue, branch-on-edit, branch-on-regenerate) wants the docker stack. Both layers matter — the splitter is too central to ship gated only on integration tests.
2. **Fix Issue 1 + Issue 3 together** via the shared compacted-prefix split step. After the split runs, reconcile only sees the raw-window suffix; Case A vs Case B is decided up front by whether the prefix matched.
3. **Fix Issue 4** — surface-specific apply policy. This unblocks proper testing of regenerate flows and naturally pulls in Issue 7 (tombstone re-keying lives in the Slack apply override).
4. **Fix Issue 8** by lifting the shared `_compute_lift_plan` internals so the Case A branch path and the compactor's lift path share `_active_paths_in_turns` / `_first_bot_with_file_activity`.
5. **Fix Issue 2.** The guardrail needs the DAG-walking ES query, which is small; it's blocker only because it's a security-shaped issue, not a complexity issue.
6. **Fix Issue 5 and Issue 6** in parallel — both are migration follow-throughs unrelated to the syncer.

After all eight: re-run `uv run ruff check`, the existing 66 unit tests, the new fake-Redis/fake-Search unit tests, the new Tier-B suite, the new `search_init.py` bootstrap, and `python -m scripts.cli "..."` against the overlay path — that last command is the simplest end-to-end smoke test that Issue 5 is closed.
