# Test Migration Plan

When the unified-conversation overlay is applied, several upstream test files
break because they reference removed or renamed types (`ContextPartition`,
`ContextPartitionItem`, `partition_uuid`, `PartitionSyncer`, `PartitionCompactor`,
`context_partition` Redis keys, etc.). The overlay's own tests cover the
syncer, models, projection, reconcile, lift plan, prefix split, branch/apply
policy, tombstone re-keying, lifted-state recompute, assistant-message
guardrail, and the eval harness — but several behavior areas have no overlay
equivalent and would lose coverage if simply deleted.

This document specs the migration of each upstream test file that doesn't
overlap with overlay tests, so the apply doesn't silently shrink the coverage
surface. Each entry names:

- The upstream file path
- What's covered today
- What needs to change to point at the new model
- The new file's intended path under `tests/unit_tests/` or `tests/integration_tests/`
- Test-by-test migration notes (with priority: P0 = must migrate, P1 = should,
  P2 = nice-to-have)
- Estimated effort

Migration order is roughly by priority and by how many other migrations depend
on it (e.g. the LLM-client fakes are foundational — port them first).

---

## 0a. Foundation: extend `FakeSearchClient.store_conversation_doc` with boundary fields

**Overlay state:** [`tests/unit_tests/_fakes.py`](../overlay/tests/unit_tests/_fakes.py) — `FakeSearchClient.store_conversation_doc(conversation, *, is_compacted, summary)` does not currently accept or persist `boundary_hash`, `boundary_message_count`, `boundary_user_count`, or `tail_hash`. These fields are written by the real compactor (`compaction.py::_compact_conversation` lines 134-146) when marking the parent compacted, and are read by `_rebuild_from_chain` (`conversation_sync.py:620-651`) to validate which compacted ancestor a client's payload corresponds to.

**Impact:** §4's chain-rebuild migration cannot pass without these fields. §3's CAS-swap migration also exercises the parent-update path and needs the same fields to be settable on stored docs (for setup) or assertable (for verifying the parent update wrote them correctly).

**Fix:**

```python
def store_conversation_doc(
    self,
    conversation: Conversation,
    *,
    is_compacted: bool = False,
    summary: str | None = None,
    boundary_hash: str | None = None,
    boundary_message_count: int | None = None,
    boundary_user_count: int | None = None,
    tail_hash: str | None = None,
) -> None:
    doc = self._build_conversation_doc(conversation, compaction_state="committed")
    doc["is_compacted"] = is_compacted
    doc["summary"] = summary
    if boundary_hash is not None:
        doc["boundary_hash"] = boundary_hash
    if boundary_message_count is not None:
        doc["boundary_message_count"] = boundary_message_count
    if boundary_user_count is not None:
        doc["boundary_user_count"] = boundary_user_count
    if tail_hash is not None:
        doc["tail_hash"] = tail_hash
    self.conversations[conversation.snapshot_uuid] = doc
```

**Effort:** ~15 minutes. Prerequisite for §3 and §4.

---

## 0. Foundation: port the LLM-client fakes

**Upstream:** [`tests/integration_tests/fakes.py`](../../../../tests/integration_tests/fakes.py) — `FakeAnthropicClient` / `FakeOpenAIClient`. Also informs the inline fakes in `test_anthropic_v1.py` / `test_openai_v1.py`.

**What's covered:** scriptable fake LLM clients that emit NDJSON at the harness contract, retain provider-specific event ordering for tool-call rounds, and reproduce every side-effect the harness relies on (`on_usage`, `ContextPartition` mutation, tool dispatch).

**Migration scope:** large — this is a prerequisite for §1, §5, and every Tier-B test. The fakes today accept `context_partition: ContextPartition` and append `ContextPartitionItem`s to its `items` list. Under the new model:

- `stream_turn` signature changes to `(items: list[ProjectedItem], instruction: str | None, model, ...)` with `on_committed_turn_item` / `on_final_assistant_message` callbacks instead of partition mutation.
- The fake must produce `TurnItem`s into `on_committed_turn_item` (function_call / function_call_output only — never intermediate assistant text) and the final assistant text into `on_final_assistant_message`.
- `summary_create_calls` / `stream_context_partitions` recording hooks need new equivalents: track `complete()` calls (carrying `items` + `instruction`) and recording the input list per `stream_turn` invocation.

**New file:** `tests/unit_tests/_llm_fakes.py` (shared helper, per `tests/CLAUDE.md` convention — fakes used in more than one test file move into a shared helper module).

**Suggested shape:**

```python
@dataclass
class LLMScript:
    rounds: list[LLMRound]
    summary_delay: float = 0.0
    summary_text: str = "STUB SUMMARY"
    think_text: str = "STUB THINK ANALYSIS"


@dataclass
class LLMRound:
    input_tokens: int = 1000
    output_tokens: int = 200
    stop_reason: str = "end_turn"
    text_deltas: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallSpec] = field(default_factory=list)


class FakeAnthropicClient:
    def __init__(self) -> None:
        self._script: LLMScript | None = None
        self._round_cursor = 0
        # Recording hooks.
        self.complete_calls: list[dict[str, Any]] = []
        self.stream_turn_calls: list[dict[str, Any]] = []

    def install_script(self, script: LLMScript) -> None: ...
    def reset(self) -> None: ...

    async def complete(
        self, items: list[ProjectedItem], instruction: str | None,
        model: str, reasoning_effort: str | None = None,
    ) -> str:
        self.complete_calls.append({"items": list(items), "instruction": instruction})
        return self._script.summary_text  # or think_text depending on context

    async def stream_turn(
        self, *, items, instruction, model, on_committed_turn_item, on_final_assistant_message,
        on_usage, max_tool_call_rounds=None, reasoning_effort=None, stream_ndjson=False,
        tool_callbacks=None,
    ):
        self.stream_turn_calls.append({"items": list(items), "instruction": instruction})
        for round_spec in self._script.rounds:
            # Emit deltas as `{"text_delta": ...}` ndjson if requested
            # For each tool_call in this round, build a TurnItem(type="function_call"),
            # call on_committed_turn_item(fc), dispatch via tool_callbacks, call
            # on_committed_turn_item(result).
            # On final round, call on_final_assistant_message(joined_text).
            ...
            if on_usage: on_usage(round_spec.input_tokens, round_spec.output_tokens)
            if stream_ndjson:
                yield json.dumps({"context_pct": ...}) + "\n"
```

`FakeOpenAIClient` mirrors the same contract using OpenAI's `ResponseStreamEvent`-shaped events.

**Effort:** ~2–3h. Foundational. Without it, §1 / §5 / Tier-B can't migrate.

---

## 1. Provider streaming + transient-narration invariant

**Upstream:**
- [`tests/unit_tests/test_anthropic_v1.py`](../../../../tests/unit_tests/test_anthropic_v1.py) (8 tests)
- [`tests/unit_tests/test_openai_v1.py`](../../../../tests/unit_tests/test_openai_v1.py) (5 tests)

**What's covered:** the transient-narration regression guard the design doc explicitly says to preserve (`README.md` §`TurnExecution` and the doc's testing section), plus context-pct emission, tool-call round continuation, and parameter passing.

**P0 — must migrate.** This is the single most-named regression guard in the design doc.

**Per-test migration:**

| Upstream test | Priority | What needs to change |
|---|---|---|
| `test_stream_turn_no_intermediate_assistant_item_before_tool_call` | P0 | Rename to `test_intermediate_narration_not_committed`. Assert `on_committed_turn_item` is called only with `function_call` / `function_call_output` items, never `type="message"`. The recorded `committed_turn_items` list must contain zero message entries after a round emits both narration and a tool call. |
| `test_stream_turn_yields_text_and_continues_after_tool_callback` | P0 | Drive the fake through a tool-use round followed by a final-text round; assert `on_final_assistant_message` receives only the final round's text (not the intermediate round's narration). |
| `test_stream_turn_emits_progress_message_for_ndjson_tool_rounds` | P0 | Assert that mid-tool-use intermediate text emits as a `progress_message` NDJSON event but does NOT call `on_committed_turn_item`. |
| `test_stream_turn_emits_context_pct_for_ndjson_stream` | P1 | Mostly mechanical — confirm `{"context_pct": N}` events still emit at the right cadence. |
| `test_stream_turn_context_pct_includes_cached_tokens` (Anthropic-only) | P1 | Same shape, against the new fake's `cache_read_input_tokens` / `cache_creation_input_tokens` field carry-through. |
| `test_stream_turn_passes_correct_params` | P1 | Replace partition-shaped assertions with assertions over the projection's `ProjectedItem` list, the `instruction` parameter, and the tool params built from `tool_callbacks`. |
| `test_anthropic_tool_param_strips_integer_minimum_from_file_tool_schema` | P0 | Standalone — tests `ToolSpec.to_anthropic_tool_param()` schema sanitization. The upstream test references `FileTool` constructed against a context partition; switch to constructing against a `current_turn_items`-style view (`view_provider=lambda: []`). The actual schema-stripping assertions are unchanged. |
| `test_openai_tool_param_keeps_integer_minimum_in_file_tool_schema` | P0 | Same — schema test, just construct `FileTool(view_provider, workspace_root=tmp_path)` instead of partition-based. |

**New files:**
- `tests/unit_tests/test_anthropic_v1.py`
- `tests/unit_tests/test_openai_v1.py`

**Effort:** ~2h once the LLM fakes (§0) exist. The schema tool-param tests are ~30 min.

---

## 1.5. `TurnItem` annotations + wire-format exclusion

**Upstream:** [`tests/unit_tests/test_api_v1_models_annotations.py`](../../../../tests/unit_tests/test_api_v1_models_annotations.py) (7 tests)

**What's covered:**
- `test_context_partition_item_has_annotations_field_default_none` — model has the `prokaryotes_annotations` field, default `None`.
- `test_context_partition_item_no_text_preamble_field` — defensive: the old `text_preamble` field really is gone.
- `test_to_openai_input_excludes_prokaryotes_annotations` — OpenAI wire format must NOT include the internal annotations field.
- `test_to_anthropic_messages_excludes_prokaryotes_annotations` — same for Anthropic.
- `test_to_openai_input_renames_system_role_to_developer` — `role="system"` → `role="developer"` in OpenAI wire format.
- `test_to_anthropic_messages_does_not_synthesize_text_block_for_function_call` — a `function_call` item preceded by another `function_call` (not a message) shouldn't get a synthetic empty text block prepended on the Anthropic side.
- `test_annotations_round_trip_through_model_dump_json` — `prokaryotes_annotations` survives a `model_dump_json` / `model_validate_json` round-trip.

**Overlay state:**
- `TurnItem.prokaryotes_annotations` still exists ([`conversation_v1/models.py:26`](../overlay/prokaryotes/conversation_v1/models.py)).
- Wire-format translation now operates on `ProjectedItem`, which has no `prokaryotes_annotations` field at all — exclusion from the wire is structural (annotations stripped at the `_turn_items_to_projected` step in [`conversation_v1/project.py`](../overlay/prokaryotes/conversation_v1/project.py)).
- `_items_to_openai_input` renames `system` → `developer` ([`openai_v1/__init__.py`](../overlay/prokaryotes/openai_v1/__init__.py)).
- `_items_to_anthropic_messages` distinguishes function_call from message blocks ([`anthropic_v1/__init__.py`](../overlay/prokaryotes/anthropic_v1/__init__.py)).

**P1 — should migrate** the structural invariants. The exclusion from the wire is now structural rather than filtered, but guard tests are cheap and worth keeping.

**Per-test migration:**

| Upstream test | Priority | New name | What changes |
|---|---|---|---|
| `test_context_partition_item_has_annotations_field_default_none` | P0 | `test_turn_item_has_annotations_field_default_none` | Build a `TurnItem(type="function_call", call_id="x", name="t")`; assert `item.prokaryotes_annotations is None`. |
| `test_context_partition_item_no_text_preamble_field` | (drop) | — | Archaeology test about a field that hasn't existed in two model generations. No value in the new model. |
| `test_to_openai_input_excludes_prokaryotes_annotations` | P0 | `test_openai_wire_excludes_prokaryotes_annotations` | Build a `TurnItem` with annotations, project it via `project_for_llm`, run `_items_to_openai_input` (in `openai_v1/__init__.py`); assert no entry in the resulting list contains a `prokaryotes_annotations` key. |
| `test_to_anthropic_messages_excludes_prokaryotes_annotations` | P0 | `test_anthropic_wire_excludes_prokaryotes_annotations` | Same shape against `_items_to_anthropic_messages`. |
| `test_to_openai_input_renames_system_role_to_developer` | P0 | Same | Build a `ProjectedItem(role="system", content="...")`, run `_items_to_openai_input`; assert the emitted entry has `role="developer"`. |
| `test_to_anthropic_messages_does_not_synthesize_text_block_for_function_call` | P0 | Same | Build `[ProjectedItem(role="user", content="x"), ProjectedItem(type="function_call", call_id="c", name="t", arguments="{}"), ProjectedItem(type="function_call_output", call_id="c", output="y")]`; run `_items_to_anthropic_messages`; assert the assistant block contains only `{"type": "tool_use", ...}` (no empty `{"type": "text", "text": ""}` prefix). |
| `test_annotations_round_trip_through_model_dump_json` | P1 | `test_turn_item_annotations_round_trip_through_model_dump_json` | Build TurnItem with annotations, dump JSON, validate back; assert annotations survived. |
| `test_to_anthropic_messages_conversion` (from `test_api_v1_models.py`) | P0 | `test_anthropic_full_conversion` | Build a representative `list[ProjectedItem]` covering: `system` message (should be dropped from messages — instruction goes via the `system` param), `user`/`assistant` text messages, `function_call` items, `function_call_output` items, consecutive same-role coalescing. Run `_items_to_anthropic_messages`; assert the resulting `list[{role, content[]}]` matches the expected shape: alternating user/assistant role groups, `tool_use` blocks at the right place, `tool_result` blocks parented under a `user` message. This is the end-to-end provider-translation test that upstream lived in `test_api_v1_models.py`; the overlay's `_items_to_anthropic_messages` does role-grouping + block-synthesis that no single existing overlay test exercises in one shot. |
| (new — OpenAI equivalent) | P0 | `test_openai_full_conversion` | Same shape against `_items_to_openai_input`. Build a representative `list[ProjectedItem]`; assert the resulting list has: instruction emitted as a leading `{"role": "developer", ...}` entry, `system`-role projected items renamed to `developer`, `function_call` items emitted with `type=function_call` plus `call_id`/`name`/`arguments`, `function_call_output` items emitted with `type=function_call_output` plus `call_id`/`output`. Upstream didn't have an OpenAI equivalent of the Anthropic conversion test, but the translation is non-trivial and worth pinning. |

**New file:** `tests/unit_tests/test_turn_item_annotations.py` (more precisely scoped than the upstream file name).

**Effort:** ~1h.

---

## 2. Compaction status endpoint

**Upstream:** [`tests/unit_tests/test_compaction_status_handler.py`](../../../../tests/unit_tests/test_compaction_status_handler.py) (5 tests)

**What's covered:** the `/compaction-status` HTTP handler's four return-shape branches:
- lock present → `{done: False}`
- partition cache evicted → `{done: True}` (no `partition_uuid`)
- partition changed to a direct child of pending → `{done: True, partition_uuid: child_id}`
- partition unchanged → `{done: True}` (no relabel target)
- partition changed but not a direct child → `{done: True}` (no relabel target)

**Overlay state:** the handler was rewritten in `prokaryotes/web_v1/compaction.py` to read `compaction_status:{pending_snapshot_uuid}` from Redis directly (no ES round-trip on the hot path). Cases:
- lock present → `{done: False}`
- relabel target stored (non-empty string) → `{done: True, snapshot_uuid: <target>}`
- relabel target stored as empty sentinel OR no key present → `{done: True}` (no `snapshot_uuid`)

**P0 — must migrate.** Endpoint is on the critical post-compaction path; behavior matters for client UX.

**Per-test migration:**

| Upstream test | Priority | New test name | What needs to change |
|---|---|---|---|
| `test_get_compaction_status_lock_present` | P0 | `test_lock_present_returns_pending` | Replace `redis.exists("compaction_lock:...")` setup; assert `done=False`, no `snapshot_uuid`. |
| `test_get_compaction_status_partition_evicted` | P0 | `test_no_relabel_target_returns_done_without_snapshot_uuid` | Set redis to have no `compaction_status:` key; assert `done=True, snapshot_uuid=None`. |
| `test_get_compaction_status_partition_changed` | P0 | `test_relabel_target_returns_done_with_snapshot_uuid` | Set `compaction_status:{pending}` → `"new-snapshot-uuid"`; assert `done=True, snapshot_uuid="new-snapshot-uuid"`. |
| `test_get_compaction_status_partition_unchanged` | P0 | `test_empty_sentinel_returns_done_without_snapshot_uuid` | Set `compaction_status:{pending}` → `""` (sentinel); assert `done=True, snapshot_uuid=None`. |
| `test_get_compaction_status_partition_changed_without_child_uuid` | (drop) | — | Subsumed by the empty-sentinel case in the new design. The "active partition's `parent_partition_uuid` doesn't match pending" case no longer applies because the polling endpoint reads only the Redis sentinel, not the active snapshot. |

**Add (per design doc's testing section):**
- `test_long_idle_past_ttl_returns_done_without_snapshot_uuid` — simulate Redis having no key (TTL expired); assert `done=True, snapshot_uuid=None`.

**New file:** `tests/unit_tests/test_compaction_status_handler.py`

**Effort:** ~1h. The fakes already cover `FakeRedis` (used by overlay tests).

---

## 3. Compaction CAS swap details

**Upstream:** [`tests/unit_tests/test_compaction_swap.py`](../../../../tests/unit_tests/test_compaction_swap.py) (13 tests)

**What's covered:** the CAS swap step (`PartitionCompactor._compact_partition`):
- accumulates `ancestor_summaries` across generations
- advances `raw_message_start_index` by tail offset
- carries forward post-snapshot messages appended during compaction
- releases the compaction lock when `compact_fn` raises
- retries Redis swap on WATCH contention
- returns early for empty summary
- skips swap when active prefix / uuid changed in Redis
- skips swap when Redis partition missing
- boundary_hash stored on ES covers full parent prefix
- ES write retry behaviors (aborts before swap on child persist fail, parent update retry exhaustion handling, CAS-never-commits handling)

**Overlay state:** the CAS swap logic was preserved nearly intact in `prokaryotes/context_v1/compaction.py::_cas_swap_child`. Field renames (`partition_uuid` → `snapshot_uuid`, etc.), but the structural behavior is unchanged. The `compaction_status:{pending_snapshot_uuid}` write is new (also tested).

**P0 — must migrate.** CAS correctness is foundational. The overlay's `test_compaction_lift_plan.py::TestMessagesMatchPrefix` covers the prefix-equality helper but none of the CAS-level invariants.

**Per-test migration:**

| Upstream test | Priority | New name | What changes |
|---|---|---|---|
| `test_compact_partition_accumulates_ancestor_summaries_across_generations` | P0 | `test_compact_conversation_accumulates_ancestor_summaries_across_generations` | Replace `ContextPartition` setup with `Conversation`. Assert `child.ancestor_summaries == [first_summary, second_summary]` after two compactions. |
| `test_compact_partition_advances_raw_message_start_index_by_tail_offset` | P0 | `test_compact_conversation_advances_raw_message_start_index_by_tail_offset` | Mechanical type swap. |
| `test_compact_partition_carries_forward_post_snapshot_messages` | P0 | `test_compact_conversation_carries_forward_post_snapshot_messages` | Now operates on `messages` (not `items`). |
| `test_compact_partition_releases_lock_when_compact_fn_raises` | P0 | `test_compact_conversation_releases_lock_when_compact_fn_raises` | Assert `compaction_lock:` deleted in the `finally` clause after raise. |
| `test_compact_partition_retries_redis_swap_on_watch_contention` | P0 | `test_compact_conversation_retries_redis_swap_on_watch_contention` | Use FakeRedis with a programmable WATCH-failure injection. |
| `test_compact_partition_returns_early_for_empty_summary` | P0 | `test_compact_conversation_returns_early_for_empty_summary` | Mechanical. Note: the overlay's empty-summary path returns *before* writing any `compaction_status:` key ([compaction.py:96-102](../overlay/prokaryotes/context_v1/compaction.py)) — it does not write the sentinel. The polling endpoint handles this correctly because the `finally:` block deletes the lock and the missing key is treated as `done=True` with no relabel target. The test should assert: lock deleted, no `compaction_status:{pending_snapshot_uuid}` key written, no CAS swap attempted. If we ever want symmetric sentinel writes from every terminal path (sentinel on no-prep, sentinel on empty summary, sentinel on CAS-skip, sentinel on exception, real uuid on commit), that's a deliberate implementation change and warrants its own ticket — current behavior relies on the "missing key + no lock" pair to signal done. |
| `test_compact_partition_skips_swap_when_active_prefix_changed` | P0 | `test_compact_conversation_skips_swap_when_active_prefix_changed` | Now uses `Conversation.messages` for the prefix check via `_messages_match_prefix`. |
| `test_compact_partition_skips_swap_when_active_uuid_changed` | P0 | `test_compact_conversation_skips_swap_when_active_snapshot_uuid_changed` | Field rename only. |
| `test_compact_partition_skips_swap_when_redis_partition_missing` | P0 | `test_compact_conversation_skips_swap_when_redis_conversation_missing` | Field rename. |
| `test_boundary_hash_stored_on_es_covers_full_parent_prefix` | P0 | `test_boundary_hash_stored_on_es_covers_full_parent_prefix` | Two boundary writes happen during compaction. The child carries **default per-snapshot boundary fields** from `put_conversation`'s `_default_boundary_fields` ([conversations.py:90](../overlay/prokaryotes/search_v1/conversations.py)) — hashed over the child's own raw window only. The parent gets a **full-prefix boundary** via `search_client.update_conversation(parent.snapshot_uuid, boundary_hash=..., boundary_message_count=..., boundary_user_count=..., tail_hash=...)` ([compaction.py:134-146](../overlay/prokaryotes/context_v1/compaction.py)) — hashed over the full reconstructed ancestor prefix + parent's own messages. The upstream test's intent is the full-prefix invariant; assert against the parent's post-update doc: read `search.conversations[parent.snapshot_uuid]`, verify `boundary_hash == compute_boundary_hash(full_parent_prefix)` and that `boundary_message_count` matches the full prefix length. |
| `test_compact_partition_aborts_before_redis_swap_if_child_persist_fails` | P0 | Same | The retry-with-backoff path in `_retry_compaction_search_write` is preserved verbatim. Test exercises monkeypatched search-client failures. |
| `test_compact_partition_keeps_committed_child_reachable_when_parent_update_retries_exhausted` | P0 | Same | Mechanical. |
| `test_compact_partition_leaves_pending_child_when_cas_never_commits` | P0 | Same | Mechanical. |

**Add (per design doc's testing section):**
- `test_messages_json_preserved_on_compacted_parent` — after the swap, the parent's `messages_json` is still populated. Verified by reading the parent doc via `search_client.get_conversation(parent.snapshot_uuid)`.
- `test_compaction_status_written_to_redis_with_child_snapshot_uuid` — assert `compaction_status:{pending}` is set to the committed child's `snapshot_uuid` at commit time, with the expected TTL.
- `test_compact_conversation_returns_early_with_no_status_write_when_pre_tail_empty` — exercises the `if not prep.pre_tail_messages` branch ([compaction.py:89](../overlay/prokaryotes/context_v1/compaction.py)). Assert: lock deleted, no `compaction_status:` key written, no summary call made. Mirrors the empty-summary test above; together they cover both "no work to do" early-return paths.

**New file:** `tests/unit_tests/test_compaction_swap.py`

**Effort:** ~3–4h. Most tests are mechanical type-swaps but the WATCH-contention and ES-retry tests need careful fake setup.

---

## 4. Compaction chain rebuild (Tier-3 of `_load_stored`)

**Upstream:** [`tests/unit_tests/test_compaction_rebuild.py`](../../../../tests/unit_tests/test_compaction_rebuild.py) (13 tests)

**What's covered:**
- `_rebuild_from_chain` assembles two-generation ancestor summaries in order
- doesn't inject stale summary without valid boundary
- includes only summaries up to the matched ancestor
- uses deepest valid compacted ancestor (boundary_hash match)
- `_walk_partition_chain` stops on conversation_uuid mismatch / cycle / intermediate missing doc
- `stream_and_finalize` emits `partition_uuid` and `compaction_pending` in the right order
- `stream_and_finalize` skips duplicate compaction when the lock is held
- sync_context_partition flows: accepts a compacted child of client's partition / preserves ancestor summaries on edit within raw window / falls back to fresh partition on retry-before-raw-window-start
- `test_sync_context_window_with_raw_message_start_index` — pure function

**Overlay state:**
- `_walk_snapshot_chain` and `_rebuild_from_chain` are preserved in `prokaryotes/context_v1/conversation_sync.py` with field renames.
- `sync_context_partition` is renamed to `sync_conversation`. The "edit within raw window" and "retry-before-raw-window-start" cases now flow through the compacted-prefix split (Issue 1+3) — the overlay's `test_conversation_syncer.py` covers them on the Tier-1 hit path, but the Tier-3 chain-rebuild path is uncovered.
- `stream_and_finalize` is now in `HarnessBase`; the events it emits are `snapshot_uuid` (in the handshake) and `compaction_pending` — same intent, renamed fields.

**P0 — must migrate** for chain rebuild correctness. The chain-walk cycle/mismatch/missing-doc tests are critical safety nets.

**Per-test migration:**

| Upstream test | Priority | New name | What changes |
|---|---|---|---|
| `test_rebuild_from_chain_assembles_two_generation_summaries_in_order` | P0 | Same (rename internal types) | Requires §0a's `store_conversation_doc(...)` extension to set `boundary_hash` + `boundary_message_count` on each compacted ancestor. Without them `_rebuild_from_chain` rejects the ancestor at the validation step (conversation_sync.py:624). Seed two compacted ancestors with valid boundary hashes against the incoming partial; assert the rebuilt `Conversation` has `ancestor_summaries = [outer, inner]`. |
| `test_rebuild_from_chain_does_not_inject_stale_summary_without_valid_boundary` | P0 | Same | Seed an ancestor with a bogus `boundary_hash`; assert it's skipped and `ancestor_summaries=[]`. |
| `test_rebuild_from_chain_includes_only_summaries_up_to_matched_ancestor` | P0 | Same | Three-generation chain; only middle ancestor's boundary matches incoming; assert only that ancestor's summary is included. |
| `test_rebuild_from_chain_uses_deepest_valid_compacted_ancestor` | P0 | Same | Two ancestors both have matching `boundary_hash`; prefer the deeper one. |
| `test_walk_partition_chain_stops_on_conversation_uuid_mismatch` | P0 | `test_walk_snapshot_chain_stops_on_conversation_uuid_mismatch` | Field rename + same cycle-detection assertion. |
| `test_walk_partition_chain_stops_on_cycle` | P0 | `test_walk_snapshot_chain_stops_on_cycle` | Same. |
| `test_walk_partition_chain_stops_when_intermediate_partition_missing` | P0 | `test_walk_snapshot_chain_stops_when_intermediate_doc_missing` | Same. |
| `test_stream_and_finalize_emits_partition_uuid_and_compaction_pending` | P0 | `test_stream_and_finalize_emits_handshake_and_compaction_pending` | Assert handshake event is the first stream event and carries `snapshot_uuid`; `compaction_pending` event still arrives after `bot_message` when threshold is hit. |
| `test_stream_and_finalize_skips_duplicate_compaction_when_lock_held` | P0 | Same | The `nx=True` SET semantics on `compaction_lock:` are unchanged. |
| `test_sync_context_partition_accepts_compacted_child_of_client_partition` | P0 | `test_sync_conversation_accepts_compacted_child_of_client_snapshot` | Already covered conceptually by overlay's `test_conversation_sync_helpers.py::TestConversationCanFollowClient`, but exercise it through the full `sync_conversation` flow. |
| `test_sync_context_partition_edit_within_raw_window_preserves_ancestor_summaries` | P1 | (subsumed) | Overlay's `test_case_a_lifted_state.py::TestCaseABranchPreservesLifted` covers this via the prefix-split + Case A flow. **No migration needed** — but cite in the new doc to make the linkage explicit. |
| `test_sync_context_partition_retry_before_raw_window_start_falls_back_to_fresh_partition` | P1 | (subsumed) | Overlay's `test_conversation_syncer.py::TestCaseBHelperAssignsSourceIds` covers via prefix-split mismatch. Same — cite the linkage. |
| `test_sync_context_window_with_raw_message_start_index` | (subsumed) | — | This upstream test exercises the sync flow when stored has `raw_message_start_index > 0` and incoming echoes the compacted prefix + raw window — exactly the compacted-prefix-split scenario. Overlay's [`test_compacted_prefix_split.py::TestMatchingPrefixStripsCleanly`](../overlay/tests/unit_tests/test_compacted_prefix_split.py) and `test_conversation_syncer.py::TestPostCompactionAppendUsesRawSuffix` cover the equivalent behavior. **No migration needed** — cite linkage. |

**New file:** `tests/unit_tests/test_compaction_rebuild.py`

**Effort:** ~3h. The chain-walk safety tests transfer almost mechanically; the `stream_and_finalize` tests need the LLM fakes from §0.

---

## 5. FileTool

**Upstream:** [`tests/unit_tests/test_file_tool.py`](../../../../tests/unit_tests/test_file_tool.py) (~60 tests)

**What's covered:** the full FileTool behavior surface — read/create/edit/conflict, range checking, revision tracking, live-window refresh, tombstoning, redundant-read detection, concurrent-write semantics, path safety. By volume the largest single test file in the repo.

**Overlay state:** `FileTool` constructor changed from `(partition, workspace_root)` to `(view_provider, workspace_root)`. The internal helpers (`_do_read_lines`, `_do_write`, `_do_create_file`, `_find_covering_window`, `_refreshable_items`) preserve their logic; only the *source* of items changed (now from `view_provider()` instead of `partition.items`). All on-disk semantics, range validation, revision-conflict handling, and live-window refresh are unchanged.

**P0 — must migrate.** Largest coverage gap if deleted. The mechanical type swap is straightforward but the volume is real.

**Migration approach:** rather than migrate test-by-test, switch the construction idiom globally:

```python
# Old
partition = ContextPartition(conversation_uuid="c-1", items=[])
tool = FileTool(partition, workspace_root=tmp_path)
# Items live in partition.items after each call.

# New
items: list[TurnItem] = []
tool = FileTool(view_provider=lambda: items, workspace_root=tmp_path)
# After each call, the result item is in tool._pending_result_items.
# Tests that previously inspected partition.items now build the view manually:
result = await tool.call(...)
items.append(result)  # simulate harness committing the item to the view
# OR for stateful round-tripping tests, write a small `_collect_view()` helper:
def view():  # closure over the test's accumulator list
    return list(items)
```

For tests that exercise `reconcile_tracked_files`, the signature also changed:

```python
# Old
await reconcile_tracked_files(partition, workspace_root=tmp_path)

# New
await reconcile_tracked_files(items, workspace_root=tmp_path)
```

**Per-area priority:**

| Test group | Priority | Notes |
|---|---|---|
| read_lines + live window annotations + range checks (`test_read_*`, `test_read_over_cap_*`) | P0 | ~12 tests. Mechanical view-provider swap. |
| create_file (`test_create_file_*`) | P0 | ~4 tests. Mechanical. |
| write actions (replace/insert/delete + context blocks) | P0 | ~10 tests. Mechanical. |
| conflict / range error / range truncated handling | P0 | ~8 tests. Mechanical. |
| reconcile_tracked_files | P0 | ~7 tests. Signature swap (partition → items list). |
| Redundant-read detection (`_find_covering_window`) | P0 | ~7 tests. Mechanical. |
| Concurrent writes / locking | P0 | ~6 tests. Mechanical. |
| `_pending_result_items` pruning | P1 | ~1 test. The new view-provider design changes the pruning logic slightly (overlay's `_refreshable_items` drops items by `id()` once they're in the view). Verify the test matches the new behavior. |
| `_refresh_live_windows` / `render_view` pure helpers | P0 | ~5 tests. Imported from `tools_v1/file_tool/live_windows.py` and `rendering.py`. Mechanical type-swap (item is `TurnItem`, not `ContextPartitionItem`). |

**New file:** `tests/unit_tests/test_file_tool.py`

**Effort:** ~6–8h. Tedious but mechanical. Can be parallelized across the test groups.

---

## 6. Compaction provider summarization wiring

**Upstream:** [`tests/unit_tests/test_compaction_provider.py`](../../../../tests/unit_tests/test_compaction_provider.py) (4 tests)

**What's covered:**
- `test_anthropic_summarize_and_compact_includes_ancestor_summaries_in_system_string` — verifies the Anthropic summarization LLM call concatenates `ancestor_summaries` into the `system` parameter.
- `test_openai_summarize_and_compact_includes_ancestor_summaries_in_developer_message` — same for OpenAI's `developer` message.
- `test_openai_summarize_and_compact_uses_only_persisted_conversation_items` — the summarization input must come only from persisted items, not transient narration.
- `test_to_anthropic_messages_appends_ancestor_summaries_after_system_instructions` — pure helper test for message-building.

**Overlay state:** the summarization flow lives in `prokaryotes/harness_v1/web.py::WebHarness._summarize_and_compact`. It now:
1. Strips live-window bodies from pre-tail `TurnExecution`s.
2. Builds a `pre_tail_conv = snapshot.model_copy(update={"messages": prep.pre_tail_messages, "lifted_turn_items": [], "lifted_anchor_source_id": None})`.
3. Projects via `project_for_llm(pre_tail_conv, historical_turns=stripped_pre_tail_turns)`.
4. Appends an explicit summarization prompt as a final user message.
5. Calls `llm_client.complete(items=items_for_summary, instruction=None, model, reasoning_effort=None)`.

Two important shape changes from the upstream tests:

- The summarization call passes `instruction=None`. There is no system / developer message at the LLM layer.
- `project_for_llm` does not inject `ancestor_summary_block` into its output — the harness builds the instruction string for *regular* turns at `_build_instruction_parts` time, not at projection time. For summarization specifically, ancestor summaries don't need to re-enter the call at all: they're already accumulated on the parent and will be carried forward verbatim into the new compacted child's `ancestor_summaries` list. The summarization input is just the pre-tail messages projected into `ProjectedItem`s, followed by an explicit summarization prompt as the final user message.

The upstream tests' "ancestor_summaries appear in the system / developer string" assertions don't translate. The new equivalent is "ancestor_summaries are not re-fed into the summarization call's input" (i.e., absent from both instruction and projected items).

**P1 — should migrate** but with substantive reformulation. The new tests should verify:

| Behavior | Priority | New test |
|---|---|---|
| Live-window bodies are stripped before summarization | P0 | `test_summarize_strips_live_window_bodies_from_pre_tail` — seed a pre-tail `TurnExecution` with a `live` file_tool output; assert `complete_calls[0]["items"]` contains the placeholder, not the body. |
| Summarization input contains only persisted items (no transient narration) | P0 | `test_summarize_uses_only_committed_turn_items` — drive a turn that emits both transient narration and committed tool items; assert the summarization-time `items` projection contains only the committed items. |
| Final user message is the summarization prompt | P1 | `test_summarize_appends_summarization_prompt_as_user_message` — assert the last item in `complete_calls[0]["items"]` is `ProjectedItem(type="message", role="user", content=_SUMMARIZATION_PROMPT)`. |
| Summarization call passes `instruction=None` | P0 | `test_summarize_passes_no_instruction` — assert `complete_calls[0]["instruction"] is None`. Guards against accidentally re-feeding ancestor summaries into the summarization LLM call. |
| Ancestor summaries are not re-fed into the projection | P0 | `test_summarize_does_not_inject_ancestor_summaries_into_items` — seed a snapshot with non-empty `ancestor_summaries`; assert no `ProjectedItem` in `complete_calls[0]["items"]` contains the summary text (it's carried forward verbatim to the child, not re-fed). |

**Drop:** `test_to_anthropic_messages_appends_ancestor_summaries_after_system_instructions` is obsolete — under the new model, the ancestor summary block is injected into the harness's instruction string at projection-input-building time (not appended after the system message). The placement semantics are different enough that the upstream test doesn't translate cleanly. Overlay's `test_project_for_llm.py` already covers projection correctness; the instruction-string assembly happens in `WebHarness._build_instruction_parts` and is best tested at the web layer (§7).

**New file:** `tests/unit_tests/test_compaction_provider.py` (or merge into `test_compaction_swap.py` since both exercise the compactor flow; suggest keeping separate for grep-ability).

**Effort:** ~1.5h.

---

## 6.5. `strip_live_window_bodies` — live-window stripping for summarization input

**Upstream:** [`tests/unit_tests/test_compaction_file_tool_lift.py`](../../../../tests/unit_tests/test_compaction_file_tool_lift.py) lines 564–733 (9 tests, all named `test_strip_live_window_bodies_*`)

**What's covered:** the helper that the compactor calls on pre-tail `TurnExecution`s before projecting them for the summarization LLM call. Tests cover:
- replacing ordinary live windows wholesale with the placeholder
- preserving CONFLICT / ALREADY_EXISTS / RANGE_ERROR / RANGE_TRUNCATED diagnostic headers
- handling the empty current-view-marker case
- preserving edit records, tombstones, and regular messages untouched
- not mutating the input (deep copy)
- stripping inactive paths too (no path-active filter at this layer)

**Overlay state:** `strip_live_window_bodies(items: list[TurnItem]) -> list[TurnItem]` lives in [`tools_v1/file_tool/live_windows.py:127`](../overlay/prokaryotes/tools_v1/file_tool/live_windows.py) and is called from [`harness_v1/web.py:285`](../overlay/prokaryotes/harness_v1/web.py) inside `_summarize_and_compact`. The signature changed from `list[ContextPartitionItem]` to `list[TurnItem]`; the body logic is unchanged.

**P0 — must migrate.** The strip step is what prevents current file contents from fossilizing into ancestor summaries. The behavior is intricate (each diagnostic header has different preservation semantics) and not covered by any overlay test.

**Per-test migration:**

| Upstream test | Priority | New name | What changes |
|---|---|---|---|
| `test_strip_live_window_bodies_replaces_ordinary_live_window_wholesale` | P0 | Same | Mechanical: build `TurnItem`s instead of `ContextPartitionItem`s. |
| `test_strip_live_window_bodies_preserves_conflict_diagnostic_header` | P0 | Same | Mechanical. |
| `test_strip_live_window_bodies_preserves_already_exists_diagnostic_header` | P0 | Same | Mechanical. |
| `test_strip_live_window_bodies_preserves_range_error_diagnostic_header` | P0 | Same | Mechanical. |
| `test_strip_live_window_bodies_preserves_range_truncated_diagnostic_header` | P0 | Same | Mechanical. |
| `test_strip_live_window_bodies_handles_empty_current_view_marker` | P0 | Same | Mechanical. |
| `test_strip_live_window_bodies_preserves_edit_records_and_tombstones_and_messages` | P0 | `test_strip_live_window_bodies_preserves_edit_records_and_tombstones` | "Messages" no longer participate (TurnItem is function_call / function_call_output only); update accordingly. |
| `test_strip_live_window_bodies_does_not_mutate_input_partition` | P0 | `test_strip_live_window_bodies_does_not_mutate_input_list` | Type rename. |
| `test_strip_live_window_bodies_strips_inactive_paths_too` | P0 | Same | Mechanical. |

**New file:** `tests/unit_tests/test_strip_live_window_bodies.py` (split from the legacy lift-test file since the lift function is gone, but strip survives).

**Effort:** ~1.5h.

---

## 7. Web routes / auth / session

**Upstream:** [`tests/unit_tests/test_web_v1.py`](../../../../tests/unit_tests/test_web_v1.py) (17 tests)

**What's covered:**
- `test_finalize_strips_system_message_and_persists_to_redis_and_es` — verifies `finalize_turn` doesn't carry the system message into persisted state.
- `test_partition_can_follow_client_*` (4 tests) — overlapping with overlay's `test_conversation_sync_helpers.py::TestConversationCanFollowClient`. **Subsumed.**
- `test_recency_tail_items_*` (2 tests) — overlapping with overlay's `test_compaction_lift_plan.py::TestRecencyTailMessages`. **Subsumed.**
- `test_message_count_before_item_index_*` (2 tests) — covered the now-removed `_message_count_before_item_index` helper. **Obsolete** (the function no longer exists; the lift step uses different bookkeeping in `_compute_lift_plan`).
- `test_sync_context_partition_*` (3 tests) — sync flow tests. Partially covered by overlay's `test_conversation_syncer.py` (Tier-1 hits), partially by §4 (chain rebuild).
- `test_webbase_abstract_property_contracts_satisfied` — verifies `WebBase` satisfies its ABC properties.
- `test_hash_password_*` / `test_verify_password_*` — pure auth helpers; **no model dependency**, **no migration needed** (kept as-is in the "structurally-untouched" group).
- `test_get_postgres_pool_raises_without_env` / `test_get_redis_client_raises_without_env` — env wiring; **no model dependency**, kept as-is.

**P1 — should migrate** the new-behavior parts.

**Per-test migration:**

| Upstream test | Priority | New behavior |
|---|---|---|
| `test_finalize_strips_system_message_and_persists_to_redis_and_es` | P0 | Verify `finalize_turn` appends the bot's `ConversationMessage`, writes Redis + Search, persists `TurnExecution` only when tool items exist, calls `refresh_assistant_index_with` (Issue 2). |
| `test_partition_can_follow_client_*` (4) | (drop) | Subsumed by overlay's `test_conversation_sync_helpers.py`. |
| `test_recency_tail_items_*` (2) | (drop) | Subsumed by overlay's `test_compaction_lift_plan.py`. |
| `test_message_count_before_item_index_*` (2) | (drop) | Function removed. |
| `test_sync_context_partition_skips_mismatched_redis_and_uses_exact_es` | P1 | New: `test_sync_conversation_skips_mismatched_redis_and_uses_exact_es`. Tier-2 path. |
| `test_sync_context_partition_uses_redis_cache_when_no_client_uuid` | P1 | New: `test_sync_conversation_uses_redis_cache_when_no_client_uuid`. |
| `test_sync_context_partition_uses_redis_cache_when_uuid_matches` | (drop) | Already covered by overlay's `test_conversation_syncer.py`'s setup (which seeds Redis with the matching snapshot and asserts the flow succeeds without ES touch). |
| `test_webbase_abstract_property_contracts_satisfied` | P1 | Update for renamed properties; mechanical. |

**Add (per design doc and Issue 2):**
- `test_post_chat_rejects_unknown_assistant_source_id` — wire up FastAPI test client; POST with a fake assistant entry; assert 400.
- `test_post_chat_accepts_known_assistant_source_id` — pre-seed the conversation, then POST with a legitimate echoed assistant; assert 200 + stream.
- `test_post_chat_rejects_assistant_without_source_id` — POST with `role="assistant"` and no source_id; assert 400.
- `test_post_chat_rejects_assistant_with_content_mismatch` — pre-seed a bot message with a known source_id; POST with an assistant entry that uses the right `source_id` but the wrong `content`; assert 400. This pins the content-mismatch branch of `validate_assistant_messages` ([conversation_sync.py:493](../overlay/prokaryotes/context_v1/conversation_sync.py)), which the other three cases don't exercise.

**Add (migrated from `test_context_v1_surface.py`):**
- `test_context_v1_surface_is_reexported_from_web_v1` — assert identity of re-exports: `web_v1.ConversationCompactor is context_v1.ConversationCompactor`, `web_v1.ConversationSyncer is context_v1.ConversationSyncer`, `web_v1._conversation_can_follow_client is context_v1._conversation_can_follow_client`, `web_v1.get_redis_client is context_v1.get_redis_client`. The overlay's `web_v1/__init__.py` still maintains these re-exports ([line 97-107](../overlay/prokaryotes/web_v1/__init__.py)); this test pins the contract so a future refactor that drops a re-export gets caught.
- `test_webbase_inherits_new_shared_layers_without_new_abstractmethods` — assert `issubclass(WebBase, HarnessBase)`, `issubclass(WebBase, CompactionStatusHandler)`, `WebBase.__abstractmethods__ == frozenset()`, and `WebBase("scripts/static")` instantiates without raising. Same intent as upstream's matching test, against the renamed inheritance chain.

**New file:** `tests/unit_tests/test_web_v1.py`

**Effort:** ~3h (most of which is FastAPI test-client setup; the assertions are mechanical).

---

## 8. Search ES wiring

**Upstream:** [`tests/unit_tests/test_search_v1_context_partitions.py`](../../../../tests/unit_tests/test_search_v1_context_partitions.py) (11 tests)

**What's covered:**
- `_extract_message_content` content joining
- `find_partition_by_tail_hash` filtering
- `partition_from_doc` parsing (4 sub-cases)
- `put_partition` indexing (with/without pending compaction metadata)
- `search_partitions`
- `update_partition`

**Overlay state:** Renamed to `prokaryotes/search_v1/conversations.py`. Helpers renamed (`partition_from_doc` → `conversation_from_doc`, `messages_from_doc`, `lifted_turn_items_from_doc`, `turn_execution_from_doc`). Method signatures unchanged in shape but operate on the new types. New methods added: `find_all_conversation_docs` (for the assistant guardrail) and `rekey_turn_execution` (for tombstone re-key).

**P1 — should migrate** the structural tests. Most can transfer mechanically.

**Per-test migration:**

| Upstream test | Priority | New name | Notes |
|---|---|---|---|
| `test_extract_message_content_joins_user_and_assistant_messages` | P0 | `test_extract_message_content_joins_messages` | `_extract_message_content` now takes a `list[ConversationMessage]`; assert it joins non-deleted content with spaces. |
| `test_find_partition_by_tail_hash_filters_to_compacted_conversation_docs` | P0 | `test_find_conversation_by_tail_hash_filters_to_compacted_docs` | Mechanical. |
| `test_items_from_doc_returns_empty_list_when_items_json_absent` | (drop) | — | `items_json` field is gone (replaced by `messages_json` + `lifted_turn_items_json` + `turn-executions.items_json`). |
| `test_partition_from_doc_returns_none_for_missing_items_json` | P0 | `test_conversation_from_doc_returns_none_for_missing_messages_json` | Mechanical. |
| `test_partition_from_doc_returns_none_for_wrong_conversation` | P0 | `test_conversation_from_doc_logs_and_returns_none_for_wrong_conversation` | Note: the overlay's helper now returns `None` (with a warning log) — it doesn't raise. Assertion needs to match. |
| `test_partition_from_doc_returns_partition` | P0 | `test_conversation_from_doc_returns_conversation` | Mechanical, but verify lifted_turn_items / lifted_anchor_source_id round-trip. |
| `test_partition_from_doc_skips_conversation_check_when_key_absent` | P0 | `test_conversation_from_doc_skips_conversation_check_when_key_absent` | Mechanical. |
| `test_put_partition_indexes_document_with_boundary_fields` | P0 | `test_put_conversation_indexes_document_with_boundary_fields` | Verify boundary_hash, tail_hash, lifted_turn_items_json, messages_json. |
| `test_search_partitions_returns_sources` | P0 | `test_search_conversations_returns_sources` | Mechanical. |
| `test_put_partition_accepts_pending_compaction_metadata` | P0 | `test_put_conversation_accepts_pending_compaction_metadata` | Mechanical. |
| `test_update_partition_sets_dt_modified` | P0 | `test_update_conversation_sets_dt_modified` | Mechanical. |

**Add (per Issues 2 & 7):**
- `test_find_all_conversation_docs_returns_all_in_dag` — seed two compacted ancestors + a branch sibling; assert all three returned. The overlay's `find_all_conversation_docs` ([conversations.py:206](../overlay/prokaryotes/search_v1/conversations.py)) returns committed-or-legacy docs only (pending compactions are excluded); the test should reflect that filter explicitly.
- `test_rekey_turn_execution_moves_doc` — seed a TurnExecution at `old_id`; `rekey_turn_execution(old_id, new_id)`; assert old gone, new present with same items, `bot_message_source_id` updated.
- `test_rekey_turn_execution_silent_no_op_on_missing_old_id` — call `rekey_turn_execution` with an old_id that doesn't exist; assert no exception, no put.

**New file:** `tests/unit_tests/test_search_v1_conversations.py`

**Effort:** ~2h.

---

## 8.5. `EvalHarness.run_task` behavior (separate from `count_turns`)

**Upstream:** [`tests/unit_tests/test_eval_harness.py`](../../../../tests/unit_tests/test_eval_harness.py) — 18 tests total. 4 are `test_count_turns_*` (overlay's same-named file already covers the rewritten `count_turns`). The other **14 tests cover `run_task` behavior** and have no overlay equivalent.

**What's covered (run_task layer):**
- `test_setup_files_written_before_agent` — fixture setup files materialize in the workspace before the agent runs.
- `test_setup_command_runs_before_agent` — `task.setup_command` runs after setup files but before the agent.
- `test_agent_receives_prompt_and_cwd` — `ScriptHarness.run(task=prompt, cwd=workspace)` plumbing.
- `test_agent_exception_recorded_as_error` — agent raises → captured in `EvalResult.error` (with full traceback).
- `test_timeout_recorded_as_error` — `asyncio.wait_for` timeout → `EvalResult.error` records the timeout.
- `test_check_files_invisible_to_agent_but_present_at_check_time` — `task.check_files` are written *after* the agent finishes and *before* the check command runs.
- `test_check_command_runs_in_workspace` — `task.check_command` runs in the per-task workspace.
- `test_check_pass_recorded` / `test_check_fail_recorded` — check command exit status drives `EvalResult.passed`.
- `test_context_partition_written_to_workspace` — the per-task artifact is persisted.
- `test_context_partition_not_written_when_none` — no artifact when the agent returns None.
- `test_tool_call_count_from_partition` / `test_tool_call_count_zero_when_no_partition` — `EvalResult.tool_call_count` derives from agent output.
- `test_think_call_count` — `EvalResult.think_call_count` derives from agent output, filtered to `name == "think"`.

**Overlay state:** the overlay's `EvalHarness.run_task` was rewritten (`harness_v1/eval.py` lines 113-184) to consume `ScriptRunResult` instead of `ContextPartition`. The artifact shape changed from `context_partition.json` to `conversation.json` + `turn_execution.json` (the latter only present when tool calls happened). Tool/think counts derive from `result.turn_execution.items` instead of `partition.items`.

**P0 — must migrate.** This is the run_task behavior layer; the overlay's own test_eval_harness only covers count_turns.

**Per-test migration:**

| Upstream test | Priority | New name | What changes |
|---|---|---|---|
| `test_setup_files_written_before_agent` | P0 | Same | The `FakeScriptHarness` upstream uses needs to return a `ScriptRunResult` instead of `ContextPartition`. Update the fake; assertions about file timing are unchanged. |
| `test_setup_command_runs_before_agent` | P0 | Same | Mechanical (FakeScriptHarness shape). |
| `test_agent_receives_prompt_and_cwd` | P0 | Same | Mechanical. |
| `test_agent_exception_recorded_as_error` | P0 | Same | Mechanical. |
| `test_timeout_recorded_as_error` | P0 | Same | Mechanical. |
| `test_check_files_invisible_to_agent_but_present_at_check_time` | P0 | Same | Mechanical. |
| `test_check_command_runs_in_workspace` | P0 | Same | Mechanical. |
| `test_check_pass_recorded` / `test_check_fail_recorded` | P0 | Same | Mechanical. |
| `test_context_partition_written_to_workspace` | P0 | `test_conversation_and_turn_execution_artifacts_written_to_workspace` | Behavior changed: assert *both* `conversation.json` and `turn_execution.json` exist when the run produced tool calls; assert `turn_execution.json` is absent when the run was pure text. |
| `test_context_partition_not_written_when_none` | P0 | `test_artifacts_not_written_when_agent_returns_none` | Mechanical (FakeScriptHarness returns None). |
| `test_tool_call_count_from_partition` | P0 | `test_tool_call_count_from_turn_execution` | Count derives from `result.turn_execution.items` filtered to `type == "function_call"`. |
| `test_tool_call_count_zero_when_no_partition` | P0 | `test_tool_call_count_zero_when_no_turn_execution` | `result.turn_execution is None` → count is 0. |
| `test_think_call_count` | P0 | Same | Filter on `type == "function_call" and name == "think"`. |

**FakeScriptHarness migration:** the upstream `FakeScriptHarness` (a local helper inside `test_eval_harness.py`) builds a fake that returns a `ContextPartition`. The new fake returns a `ScriptRunResult`:

```python
@dataclass
class FakeScriptRun:
    items: list[TurnItem] = field(default_factory=list)
    final_assistant_text: str = "ok"

class FakeScriptHarness:
    def __init__(self, ...):
        self.calls: list[dict] = []
        self.return_value: FakeScriptRun | None = FakeScriptRun()

    async def run(self, *, task, cwd, max_tool_call_rounds, on_usage, verbose):
        self.calls.append({"task": task, "cwd": cwd})
        if self.return_value is None:
            return None
        conv = Conversation(conversation_uuid="fake", bot_author_id="__bot__", messages=[])
        return ScriptRunResult(
            conversation=conv,
            final_assistant_text=self.return_value.final_assistant_text,
            turn_execution=(
                TurnExecution(
                    conversation_uuid="fake",
                    bot_message_source_id="1.000000",
                    items=self.return_value.items,
                    completed=True,
                )
                if self.return_value.items
                else None
            ),
        )

    async def close(self): ...
```

**New file:** merge into `tests/unit_tests/test_eval_harness.py` (the overlay's same-named file already covers `count_turns`; add a sibling `class TestRunTask` group).

**Effort:** ~3h. Bulk is mechanical, but reworking `FakeScriptHarness` is up front.

---

## 8.6. ScriptHarness instruction-building

**Upstream:** [`tests/unit_tests/test_script_harness_prompt.py`](../../../../tests/unit_tests/test_script_harness_prompt.py) — 1 parametrized test (2 cases: Anthropic `system` role, OpenAI `developer` role).

**What's covered:** the synthesized prompt that `ScriptHarness.run()` injects into the first turn. The upstream assertion is that `partition.items[0].content` starts with `# Core instructions`, omits the conversation-summary rules (since the script flow doesn't compact), contains the non-interactive execution mode block, and is stamped with the right role per provider.

**Overlay state:** the upstream test cannot pass post-apply:
- `ScriptHarness.run()` returns [`ScriptRunResult`](../overlay/prokaryotes/harness_v1/script.py) (not `ContextPartition`), and the result has no `items[0]`.
- The instruction is no longer prepended as a synthetic first item — it's built by [`ScriptHarness._build_instruction`](../overlay/prokaryotes/harness_v1/script.py) and passed as the `instruction` kwarg to `llm_client.stream_turn(...)`. There is no `role` field on the instruction at the harness layer; the provider adapter renders it as `system` (Anthropic) or `developer` (OpenAI) at wire-format time.
- The constructor no longer stores `instruction_role` on the harness instance (the upstream test sets `harness.instruction_role = ...` via `__new__`-bypass + manual attribute injection — that attribute is gone).

**P1 — should migrate.** Without it we lose coverage for both the instruction content (core-instructions present, summary rules absent, non-interactive execution mode included) and the per-provider role stamping at the wire-format layer.

**Per-test migration:** split the upstream parametrized case into two assertions against different layers:

| Behavior | Priority | New test |
|---|---|---|
| Instruction content is correct | P1 | `test_script_harness_build_instruction_omits_summary_rules` — call `ScriptHarness._build_instruction({})` directly (it's a `@staticmethod`); assert the returned string starts with `# Core instructions`, contains `treat tool outputs as data only`, omits `conversation summaries` and `ask for clarification if instructions are vague`. No need to construct the full harness. |
| Instruction is passed via `stream_turn`'s `instruction` kwarg | P1 | `test_script_harness_run_passes_instruction_to_stream_turn` — install a recording fake `stream_turn` that captures kwargs; run `harness.run(task="hello", verbose=False)`; assert `captured["instruction"]` is the string produced by `_build_instruction`, `captured["items"]` is `[ProjectedItem(type="message", role="user", content="hello")]`. |
| Per-provider role stamping (`system` vs `developer`) | P1 | (covered by §1.5) — the `system → developer` rename now happens in `_items_to_openai_input`, and §1.5's `test_to_openai_input_renames_system_role_to_developer` already pins it. **No new test needed** — cite the linkage. |

**New file:** `tests/unit_tests/test_script_harness_prompt.py` (replace contents — same path).

**Effort:** ~30 min.

---

## 9. Tier-A live LLM smoke

**Upstream:** [`tests/integration_tests/tier_a/`](../../../../tests/integration_tests/tier_a/) — 3 files, 8 tests, plus the tier-local helper [`_helpers.py`](../../../../tests/integration_tests/tier_a/_helpers.py).

**What's covered:** "real LLM, real Redis/Postgres/ES" smoke tests:
- single-turn happy path (Anthropic + OpenAI)
- forced compaction (Anthropic + OpenAI)
- tool call best effort (Anthropic + OpenAI)
- memory continuity across compaction (judged by GPT-4)
- branch isolation after retry-before-tail (judged by GPT-4)

**Migration scope:** larger than fixture-reuse from §10. Tier A does **not** share its helpers with Tier B — it has its own [`tier_a/_helpers.py`](../../../../tests/integration_tests/tier_a/_helpers.py) (`post_chat_collect`, `wait_for_compaction`, `drive_to_compaction`) that hard-codes old-model APIs in three places that all need migration:

1. **Tier-A helper rewrite** — `_helpers.py` imports `ContextPartition`, extracts `partition_uuid` from `events[0]["partition_uuid"]`, queries `/compaction-status` with `pending_partition_uuid`, and parses `ContextPartition.model_validate_json(cached)` out of `redis:context_partition:{conv_uuid}`. The same `TurnRecord`-shaped rewrite §10 prescribes for Tier B applies verbatim: read `snapshot_uuid` from the handshake event, capture `source_id_assignments` and the `bot_message` `source_id` for echo on subsequent POSTs, poll `/compaction-status` with `pending_snapshot_uuid`, and parse `Conversation.model_validate_json(cached)` out of `redis:conversation:{conv_uuid}`.
2. **Smoke test bodies** — all three of `test_smoke_anthropic.py`, `test_smoke_openai.py`, `test_smoke_judged.py` directly `from prokaryotes.api_v1.models import ContextPartition`, assert `types[0] == "partition_uuid"`, call `web_harness.search_client.get_partition(...)` (renamed to `get_conversation`), and dereference `partition.partition_uuid` / `parent_partition_uuid`. Mechanical but real: import swap, first-event-shape rewrite to `handshake`, field renames throughout.
3. **Judged scenarios** — `test_smoke_judged.py`'s "memory continuity across compaction" and "branch isolation after retry-before-tail" tests reference compaction branch identity via `partition_uuid` / `parent_partition_uuid`. Update to the new branch / snapshot semantics (relabel target via `/compaction-status` polling rather than `partition_uuid` succession).

**P1 — should migrate.** Tier A is the closest thing we have to an integration-level regression guard.

**Approach:** rewrite `_helpers.py` first against the new wire (mirrors §10's `TurnRecord` shape), then mechanically migrate the three smoke files (~30–50 lines each) on top of the new helpers. Tier A does not literally reuse Tier-B helpers — keep them tier-local.

**Effort:** ~4–5h (helper rewrite ~2h, three smoke files ~30 min each plus the judged-test reformulation).

---

## 10. Tier-B integration suite

**Upstream:** [`tests/integration_tests/tier_b/`](../../../../tests/integration_tests/tier_b/) — 5 files, ~28 tests. Plus [`conftest.py`](../../../../tests/integration_tests/tier_b/conftest.py) and [`tests/integration_tests/fakes.py`](../../../../tests/integration_tests/fakes.py).

**What's covered:**
- `test_auth_boundary.py` — chat rejects missing session / empty messages.
- `test_chat_flow.py` — single-turn happy path, multi-turn continuation, tool-call round trip, think-tool round trip.
- `test_compaction_flow.py` (8 tests) — forced compaction, mid-summary message carry-forward, multi-generation ancestors, Redis-miss rebuild, multi-generation rebuild after miss, retry-within-tail, retry-before-tail (Case B), branch switch.
- `test_file_tool_flow.py` (9 tests) — live-window survival across writes / compaction / file_tool diagnostic edge cases.
- `test_stream_protocol.py` (4 tests) — `partition_uuid` first event, `context_pct` ordering, tool-round event ordering, `compaction_pending` last.

**Migration scope:** large. Depends on §0 (LLM fakes), §5 (FileTool ported), §7 (web routes wired through `validate_assistant_messages`).

**Tier-B-specific concerns:**

- The conftest's `_web_harness_anthropic` / `_web_harness_openai` fixtures replace `harness.llm_client` with `FakeAnthropicClient()` / `FakeOpenAIClient()` **before `init()`**. This pattern transfers verbatim once the new fakes (§0) are in place.
- `tests/integration_tests/env_bootstrap.py` sets env vars before `prokaryotes` imports — also transfers verbatim (the env vars are still `COMPACTION_TOKEN_THRESHOLD_PCT`, `COMPACTION_RECENCY_TAIL`, etc.).
- The `_authed_client` flow (`POST /register`, then re-use the cookie) is unchanged.

**Per-file priority:**

| File | Priority | Notes |
|---|---|---|
| `test_auth_boundary.py` | P0 | 2 tests, mechanical — no model interactions. |
| `test_chat_flow.py` | P0 | 4 tests. Mechanical once fakes are ported. Add: `test_post_rejects_unknown_assistant_source_id` (overlay's Issue 2 guardrail). |
| `test_compaction_flow.py` | P0 | 8 tests. The richest behavior coverage. `test_retry_before_recency_tail` is essential — it's the Case B integration test the review doc named (`test_unified_web_flow.py::test_compaction_relabel_and_continue` in the doc). |
| `test_file_tool_flow.py` | P0 | 9 tests. Mechanical once §5 is done. |
| `test_stream_protocol.py` | P0 | 4 tests. Not mechanical — the stream shape itself changed. The old `partition_uuid` first event is replaced by a `handshake` event (`{snapshot_uuid, source_id_assignments[], unacknowledged_bot_messages?[]}`). A new `bot_message` event is the last persistence-relevant event. Existing assertions on first-event identity and stream ordering need to be rewritten against the new event vocabulary. See "Stream-protocol migration depth" below. |

**Add (per the apply's review-doc reference to `test_unified_web_flow.py`):**
- `test_compaction_relabel_and_continue` (in `test_compaction_flow.py`): full round-trip — new conversation → repeated turns until compaction → relabel via `/compaction-status` → next-turn projects the bot's full history exactly once (no duplication from Issue 1's old failure mode).

**Stream-protocol migration depth.** The existing Tier-B helpers ([`_run_turn`, `_run_until_compaction` etc. in `test_compaction_flow.py`, `test_stream_protocol.py`, `test_chat_flow.py`](../../../../tests/integration_tests/tier_b/)) extract `partition_uuid` from the first stream event and thread it into the next POST as `payload["partition_uuid"]`. Under the new wire that pattern is wrong in three ways and needs a single shared helper rewrite — don't try to patch each call site:

1. **First event is now a handshake** with shape `{"snapshot_uuid": "...", "source_id_assignments": [...], "unacknowledged_bot_messages": [...] (optional)}`. The helper has to:
   - Read `snapshot_uuid` from the handshake (not from a separate top-level `partition_uuid` field).
   - Apply `source_id_assignments` to the local list of user messages it just submitted, so subsequent POSTs can include the assigned `source_id` per message. Tests that submit `[user_msg]` and don't track the assigned id will fail to echo it on the next POST.
2. **`bot_message` event marks the final commit.** The assistant's full text is reconstructible from `text_delta` events, but its identity (its server-assigned `source_id`) only arrives on the `bot_message` event. Helpers must:
   - Wait for `{"bot_message": {"source_id": "..."}}` before treating the turn as done.
   - Capture that `source_id` and include the assistant entry on subsequent POSTs (`{"role": "assistant", "content": <accumulated text>, "source_id": <captured>}`). Otherwise the assistant-message guardrail (issue 2) rejects the next POST with a 400 because the echoed content has no known `source_id`.
   - Treat the absence of `bot_message` (mid-turn abort, max-rounds-hit) as "no assistant node should be created" — tests that previously asserted "stream ended cleanly" need to distinguish committed-vs-aborted.
3. **Polling endpoint param renamed.** `/compaction-status` now takes `pending_snapshot_uuid` (not `pending_partition_uuid`) and returns `{"done": bool, "snapshot_uuid": str?}` (not `{"done": bool, "partition_uuid": str?}`). The relabel logic is the same idempotent walk; only the names change. The legacy side-channel clear ("subsequent stream handshake with a different id clears the indicator") is **removed** — polling is the only clear path.

Suggested approach: rewrite `_run_turn` / `_run_until_compaction` to return a small `TurnRecord` carrying `snapshot_uuid`, `source_id_assignments`, `bot_message_source_id` (or `None` for aborted), and the captured text. Existing per-test extraction logic largely disappears.

**New location:** `tests/integration_tests/tier_b/` (same structure).

**Effort:** ~6–8h. Bulk of the migration work overall.

---

## 11. UI / JS tests

**Upstream:** [`tests/ui_tests/ui.test.js`](../../../../tests/ui_tests/ui.test.js) (DOM-based tests for `scripts/static/ui.js`) and [`tests/ui_tests/file_tool_ui.test.js`](../../../../tests/ui_tests/file_tool_ui.test.js).

**What's covered today:**
- `ui.test.js` — message tree navigation, edit/regenerate flows, fork navigation, compaction-pending indicator, polling-driven relabel, side-channel clear on handshake. 9 hits on `partition_uuid` / `partitionUuid` / `relabelPartitionUuid` across the file.
- `file_tool_ui.test.js` — file-tool UI affordances. 0 hits on partition vocabulary.

**Overlay state:**
- The overlay added [`tests/ui_tests/conversation_client.test.js`](../overlay/tests/ui_tests/conversation_client.test.js) (12 tests) covering the new pure module `scripts/static/conversation_client.js` — `applyHandshake`, `applyBotMessage`, `relabelSnapshotUuid`, `applyResyncHandshake`, `buildRequestMessages`. These are pure-function tests, no DOM.
- The actual `scripts/static/ui.js` (DOM-aware, fetch-aware) has not been migrated. Its tests still use `partition_uuid` vocabulary and the old wire-event shapes (`partition_uuid` first event, no `bot_message`, no `source_id_assignments`).
- Per the overlay's `README.md` integration note, the wire-up itself is straightforward (rename `partitionUuid` → `snapshotUuid` throughout, route handshake/bot-message events through the new client primitives, remove the legacy side-channel clear). But the existing tests don't exercise the new wire and would all fail if the apply lands without a `ui.js` update.

**P0 — must migrate.** UI is the user-facing surface; broken JS tests block any frontend regression coverage.

**Per-area migration:**

| Test group | Priority | What changes |
|---|---|---|
| **Stream-handshake parsing** — `partition_uuid` first event | P0 | The new wire emits a handshake event as the first stream event, shape `{snapshot_uuid, source_id_assignments[], unacknowledged_bot_messages?[]}`. Update assertions accordingly. Add tests for `source_id_assignments` mapping client-side nodes. |
| **`bot_message` event** | P0 | The new wire emits `{"bot_message": {"source_id": ...}}` as the last persistence-relevant event. The client creates the assistant node only on this event (not from accumulated `text_delta` alone). Add tests verifying: assistant node *not* created if stream aborts before `bot_message`; assistant node carries both `source_id` and `snapshot_uuid` after `bot_message`. |
| **Compaction-pending indicator + relabel** | P0 | Three coordinated renames on the request and response sides: (a) `messageTree` node field `partitionUuid` → `snapshotUuid`; (b) `/compaction-status` request query param `pending_partition_uuid` → `pending_snapshot_uuid` ([`web_v1/compaction.py:24`](../overlay/prokaryotes/web_v1/compaction.py)) — the UI's fetch URL needs updating; (c) response body field `partition_uuid` → `snapshot_uuid` (now `{done, snapshot_uuid?}`). `relabelSnapshotUuid` (overlay's idempotent walk) is the only path that updates labels. The legacy side-channel clear (on any subsequent stream handshake with a different id) is **removed** — switching to a sibling branch and sending must NOT clear an unrelated pending indicator. Add the anti-case: branch switch during pending compaction → indicator stays, indicator only clears on its branch's `/compaction-status` poll returning `done`. |
| **Resync handshake** (stream-loss recovery) | P0 | New behavior. POST that the server detects as a post-commit drop returns a handshake with `unacknowledged_bot_messages`. Add tests for: send-from-leaf compose mode (reparent pending user under recovered bot, auto-retry); edit/regenerate compose mode (pop pending user, restore content as draft, no auto-retry); chained-trailing-bot reconstruction. |
| **Assistant-message guardrail surfacing** | P1 | The web route now returns 400 on `unknown assistant source_id` / `content mismatch`. UI tests should verify the error surfaces in the UI (probably as an inline error or toast) and the pending user node behavior is reasonable. |
| **`buildRequestMessages` wiring** | P0 | UI tests need to verify `sendMessage` builds the request via `buildRequestMessages(messageTree, activePath)` — emitting `source_id` only on server-stamped nodes. |

**Approach:** rather than rename inside `ui.test.js`, do a clean-slate rewrite targeting the new wire and the new client primitives. Keep `file_tool_ui.test.js` as-is (no model dependency).

**New files:**
- `tests/ui_tests/ui.test.js` — rewritten against new wire + client primitives.
- (existing) `tests/ui_tests/conversation_client.test.js` — pure-module tests, unchanged.
- (existing) `tests/ui_tests/file_tool_ui.test.js` — unchanged.

**Effort:** ~4–5h. Sizable because the upstream `ui.test.js` is the largest JS test file and exercises a lot of behavior. Could be parallelized across the test groups above.

---

## Summary table

| Area | Priority | Effort | Depends on |
|---|---|---|---|
| §0a Fake helper boundary fields | P0 | 15 min | — |
| §0 LLM-client fakes | P0 | 2–3h | — |
| §1 Provider streaming + transient narration | P0 | 2h | §0 |
| §1.5 TurnItem annotations + wire exclusion | P1 | 1h | — |
| §2 Compaction status endpoint | P0 | 1h | — |
| §3 Compaction CAS swap | P0 | 3–4h | §0a |
| §4 Compaction chain rebuild | P0 | 3h | §0, §0a |
| §5 FileTool | P0 | 6–8h | — |
| §6 Compaction provider summarization | P1 | 1.5h | §0 |
| §6.5 `strip_live_window_bodies` | P0 | 1.5h | — |
| §7 Web routes / auth / session | P1 | 3h | §0 |
| §8 Search ES wiring | P1 | 2h | §0a |
| §8.5 EvalHarness run_task | P0 | 3h | — |
| §8.6 ScriptHarness instruction | P1 | 30 min | — |
| §9 Tier-A live LLM smoke | P1 | 4–5h | §10 (wire vocab), not helper reuse |
| §10 Tier-B integration suite | P0 | 6–8h | §0, §5, §7 |
| §11 UI / JS tests | P0 | 4–5h | — |

**P0 total:** ~32–40h of focused work.
**P1 total:** ~11–13h additional.

Suggested order: §0a (15 min, unblocks §3 §4 §8) → §0 → §2 / §3 / §6.5 / §8 / §8.5 / §8.6 / §11 in parallel (no shared deps beyond §0a) → §1 / §1.5 / §4 / §5 / §6 / §7 → §10 → §9.

---

## What to delete outright (no migration needed)

These files reference removed types and have no overlap with overlay tests OR are entirely subsumed:

- `tests/unit_tests/context_partition_utils.py` — old-model test helper.

## Partially deletable — extract the still-relevant tests, then delete the file

- `tests/unit_tests/test_api_v1_models.py` — 7 tests, each with a clear new home; none survive in place but several have explicit destinations rather than being passively subsumed:
  - `test_ancestor_summary_block_labels_background_memory` → overlay's [`test_conversation_models.py::TestAncestorSummaryBlock`](../overlay/tests/unit_tests/test_conversation_models.py) (already migrated).
  - `test_hash_helpers_are_role_and_content_sensitive` → overlay's `test_conversation_models.py::TestComputeBoundaryHash` (already migrated; the payload is now `{author_id, content}` not `{role, content}` per the design doc's clean break — assertions adjusted).
  - `test_partition_uuid_round_trip` → overlay's `test_conversation_models.py` model round-trip coverage (already exists implicitly via pydantic).
  - `test_find_context_divergence` → overlay's [`test_reconcile.py`](../overlay/tests/unit_tests/test_reconcile.py) covers the new reconcile classifier (replaces positional divergence).
  - `test_sync_context_window` / `test_sync_context_window_exception` → overlay's [`test_conversation_syncer.py`](../overlay/tests/unit_tests/test_conversation_syncer.py) end-to-end sync flow covers this.
  - **`test_to_anthropic_messages_conversion`** → **migrate to §1.5** as a new comprehensive Anthropic conversion test. See §1.5 below; this is the one row not covered by either the overlay's existing tests or §1.5's current entries. Adding it explicitly.
- `tests/unit_tests/test_api_v1_models_annotations.py` — most assertions still apply to the new model. See §1.5; migrate the relevant tests, then delete the file.
- `tests/unit_tests/test_context_v1_surface.py` — 2 tests covering the `web_v1` re-export contract (4 symbols identity-checked against `context_v1`) and `WebBase`'s ABC contract (empty abstractmethods, instantiable). Overlay's `web_v1/__init__.py` still re-exports the equivalent symbols (`ConversationCompactor`, `ConversationSyncer`, `_conversation_can_follow_client`, `get_redis_client`). See §7; add a migrated surface check there, then delete the file.
- `tests/unit_tests/test_compaction_file_tool_lift.py` — the lift helper is gone (overlay's `test_compaction_lift_plan.py` covers `_compute_lift_plan`), but `strip_live_window_bodies` survives in [`tools_v1/file_tool/live_windows.py`](../overlay/prokaryotes/tools_v1/file_tool/live_windows.py) and is still used by the summarization flow. See §6.5; migrate the strip-body tests, then delete the file.
- `tests/unit_tests/test_eval_harness.py` — the 4 `count_turns` tests are subsumed by overlay's same-named file. The other 14 tests cover `run_task` behavior (setup/check files, check commands, timeout handling, artifacts, tool/think counts) — see §8.5. Migrate `run_task` tests into the overlay's file (as a sibling `class TestRunTask` group), then delete the upstream file.

---

## What stays untouched (no model dependency)

These already pass cleanly post-apply and don't need migration:

- `tests/unit_tests/test_integration_env_bootstrap.py`
- `tests/unit_tests/test_integration_stream_utils.py`
- `tests/unit_tests/test_search_v1_topics.py`
- `tests/unit_tests/test_shell_command.py`
- `tests/unit_tests/test_system_message_utils.py`
- `tests/unit_tests/test_think.py`
- `tests/unit_tests/test_utils_v1_text_utils.py`
