# Conversation Compaction

## Overview

Provider LLM APIs impose hard context-window limits. Without intervention, a long conversation eventually exceeds the limit and subsequent requests fail. Compaction allows conversations to continue indefinitely by periodically summarizing the oldest portion of the conversation history and replacing it with a compact, LLM-generated briefing. The rest of the system — branching, retries, Redis caching, Elasticsearch persistence — continues to function correctly across compaction boundaries.

---

## Core Concepts

### Partition Chains

Before compaction, a conversation is a single flat `ContextPartition`: an ordered list of items (messages, tool calls, tool outputs) keyed by `conversation_uuid` in Redis. Compaction introduces a **chain** of partitions:
```
[P0: compacted] → [P1: compacted] → [P2: active]
```

Each partition carries a `partition_uuid` and an optional `parent_partition_uuid` that points to its predecessor. A compacted partition is sealed — its raw items are archived in Elasticsearch and it carries an LLM-generated summary of its contents. The active partition is the current working set and is the only one cached in Redis.

When the model is called, context is assembled as:
```
[core instructions] + [summary block from P0/P1] + [raw items from P2]
```

Summaries are injected into the system (Anthropic) or developer (OpenAI) message as a trailing `# Compacted conversation summary` background-memory section after the core instructions. They are not injected as conversation turns, which means the model sees them as context rather than as attributed exchanges.

### Ancestor Summaries

Each `ContextPartition` carries an `ancestor_summaries` list — the ordered (oldest-first) collection of summaries from all compacted ancestors. Because summaries are immutable once generated, they are cached in Redis alongside the raw items of the active partition. This means the chain walk through Elasticsearch is paid exactly once per branch: after a structural rebuild the partition written to Redis already includes all ancestor summaries, and subsequent requests on that branch take the Redis fast path with no further ES involvement.

### Recency Tail

When a partition is compacted, the most recent K user/assistant message turns are retained verbatim in the new active partition rather than being swept into the summary. This recency tail gives the model literal context for the exchanges most likely to be referenced in the next turn. The tail size K is configurable via `COMPACTION_RECENCY_TAIL` (default 6 messages).

The boundary between summarized history and verbatim recency tail is tracked by `raw_message_start_index` on the active partition: the integer offset into the full `ChatConversation.messages` list where the partition's raw span begins. A fresh partition always has `raw_message_start_index = 0`.

### Boundary Hashing

A compacted partition's summary is only valid if the conversation branch that produced it is provably identical to the current branch. To enforce this, each compacted partition stores a `boundary_hash` — a SHA-256 digest of the full role/content sequence of every user and assistant message covered by the summary.

When the server needs to determine whether a compacted ancestor's summary is applicable (e.g., on a retry that lands before the recency tail), it computes the same hash over the corresponding prefix of the incoming `ChatConversation.messages` and compares. A mismatch means the branches diverged before the compaction boundary and the summary must not be used. This prevents stale summaries from contaminating forked branches.

A secondary `tail_hash` (hash of the last N user message contents) is stored as a lookup key for future use but is not sufficient on its own to prove applicability.

---

## Data Model

### ContextPartition

The `ContextPartition` model gains the following fields to support compaction:

| Field | Description |
|---|---|
| `partition_uuid` | Unique identifier for this partition. Auto-generated. |
| `parent_partition_uuid` | UUID of the preceding compacted partition, or `null` for the root. |
| `ancestor_summaries` | Ordered list of LLM-generated summaries from all compacted ancestors. |
| `raw_message_start_index` | Offset into `ChatConversation.messages` where this partition's raw span begins. |

The `items` list retains its existing semantics: messages, tool calls, and tool outputs for the raw (non-summarized) portion of the conversation.

### Elasticsearch: `context-partitions` Index

Every partition is persisted to Elasticsearch. Redis is the hot cache for the currently active branch; Elasticsearch is the durable store for all partitions — active, forked, and compacted. The ES document carries:

| Field | Purpose |
|---|---|
| `partition_uuid` | Document `_id`; enables O(1) lookup by UUID. |
| `conversation_uuid` | Scopes partitions to a conversation. |
| `parent_partition_uuid` | Enables ancestor chain walking. |
| `is_compacted` | Boolean; set `true` when the compaction summary has been finalized. |
| `summary` | LLM-generated summary of this partition's content. Full-text indexed. |
| `items_json` | Serialized raw items. Stored but not indexed (excluded from the inverted index). |
| `message_content` | Extracted user/assistant text. Full-text indexed for future search. |
| `boundary_hash` | Hash of all messages covered by the summary; proves applicability. |
| `boundary_message_count` | Message count at the compaction boundary. |
| `boundary_user_count` | User-message count at the boundary (supplementary). |
| `tail_hash` | Hash of the last N user messages; secondary lookup aid. |
| `raw_message_start_index` | Mirrors the in-memory field for chain reconstruction. |
| `ancestor_summaries` | Mirrors the in-memory field for chain reconstruction. |
| `dt_created` / `dt_modified` | Timestamps. |

The `items_json` field is intentionally excluded from the ES inverted index to keep the index size manageable. Raw items are rehydrated from this field when a partition must be reconstructed from ES. Storing `message_content` separately means conversation content is searchable without parsing `items_json`.

---

## Stream Protocol

Every response stream begins with a `partition_uuid` event, followed by the normal `text_delta` events and `context_pct` events, and optionally a `compaction_pending` event at the end.

**`partition_uuid`** — Emitted first in every response stream. The client records this UUID on the assistant tree node it creates for that response. On the next request, the client sends back the `partition_uuid` of the most recent node on the currently active path. This is how the server knows which branch is active.

**`context_pct`** — An integer percentage of the model's context window consumed, emitted after each LLM round using the input token count from that round's usage data. Rendered as a fill indicator in the UI so the user can anticipate an upcoming compaction.

**`compaction_pending`** — Emitted at the end of a response when a background compaction has been scheduled. The UI shows a non-blocking, pulsing indicator. The indicator clears in two ways: (1) when a subsequent stream arrives carrying a different `partition_uuid`, confirming the Redis swap completed and a new active partition is in use; or (2) automatically via background polling — the client issues `GET /compaction-status?conversation_uuid=…&pending_partition_uuid=…` every 5 seconds and clears the indicator as soon as the response returns `{"done": true}`, without requiring the user to send another message.

On the request side, `ChatConversation` carries an optional `partition_uuid` field. The client sends back the UUID last received for the active branch, or `null` for a fresh conversation.

---

## Request Reconciliation

`sync_context_partition` in `WebBase` is responsible for producing the correct `ContextPartition` for each incoming request. It follows a strict priority order:

### 1. Redis Fast Path

Redis holds at most one partition per `conversation_uuid`. If the cached partition's `partition_uuid` matches the client's (or the client sent no UUID, or the cached partition is the direct parent of the client's UUID), the server attempts `sync_from_conversation`. This method reconciles the partition's raw items against the incoming `ChatConversation.messages` starting from `raw_message_start_index`:

- If the conversation matches the partition exactly, a `ConversationMatchesPartitionError` is raised and the partition is returned unchanged.
- If the conversation is longer, items are appended.
- If there is a divergence (retry or edit), items are truncated at the divergence point and the new message is appended.
- If the incoming message count falls below `raw_message_start_index`, a `ConversationOutsideRawWindowError` is raised and the fast path is abandoned.

If the client's `partition_uuid` does not match the cached partition and the cached partition is not the direct parent of the client's UUID, the Redis partition is stale for this branch. The server falls through to ES.

### 2. Exact ES Partition Load

If the client supplied a `partition_uuid`, the server fetches that exact document from Elasticsearch by `_id`. If the document exists and is not marked `is_compacted`, the server attempts the same sync. Compacted partitions are skipped here because they are sealed and should not be resumed directly; they are only usable via chain reconstruction.

### 3. Ancestor Chain Reconstruction

If neither the Redis nor the exact ES partition can be reconciled, the server walks the ancestor chain from the client's `partition_uuid` by following `parent_partition_uuid` links through Elasticsearch. For each compacted ancestor (oldest-first), it validates applicability by computing `boundary_hash` over the corresponding prefix of `ChatConversation.messages` and comparing against the stored value. The deepest ancestor that passes validation becomes the reconstruction anchor:

- `ancestor_summaries` are collected from all validated ancestors up to and including the anchor.
- `raw_message_start_index` is set to the anchor's `boundary_message_count`.
- `items` are populated from the conversation messages after that boundary.

If no ancestor validates (e.g., the branch diverged before any compaction boundary), the server falls back to a fresh partition built directly from the incoming messages with no ancestor summaries. Stale summaries are never injected.

After reconstruction, the new partition is written to both Redis and Elasticsearch so subsequent requests on this branch take the fast path.

---

## Compaction Flow

### Trigger

Each web harness registers an `on_usage` callback with the LLM client. After each LLM round, the client reports `input_tokens` and `output_tokens`. The callback computes the percentage of the model's context window consumed (using `MODEL_CONTEXT_WINDOWS`, a per-model lookup table) and sets a `pending_compaction` flag if the percentage meets or exceeds `COMPACTION_TOKEN_THRESHOLD_PCT` (default 80%). The flag is a request-local closure variable — it is never serialized to Redis or ES.

### Sequencing in `stream_and_finalize`

`stream_and_finalize` orchestrates the handoff after the response generator is fully drained:

1. Emit `partition_uuid` at the start of the stream.
2. Yield all model events (text deltas, context_pct, tool-call rounds).
3. If `pending_compaction` is set and the compaction lock can be acquired:
   - Await `finalize()` **synchronously** to ensure the current partition is committed to Redis before the background compactor begins its WATCH loop.
   - Emit `compaction_pending`.
   - Deep-copy the finalized partition as a snapshot.
   - Fire `_compact_partition` as a background task and return.
4. Otherwise, background `finalize()` as normal.

The synchronous `finalize()` in step 3 is a deliberate departure from the default behaviour (where finalize is backgrounded). It closes the race where a background `finalize()` SET could overwrite the result of the compaction swap.

### Compaction Lock

Before firing `_compact_partition`, the server acquires `compaction_lock:{conversation_uuid}` via Redis `SET NX` with a TTL set slightly longer than expected summary latency. If the lock is already held by a running compaction, the new trigger is skipped — the in-flight compaction will produce a new active partition that already incorporates the current state. The lock is released unconditionally in the `finally` block of `_compact_partition`.

### `_compact_partition`

The background task performs the following steps:

1. **Write the snapshot to ES** with `is_compacted=False`. This crash-recovery marker ensures that an interrupted compaction leaves a recoverable document that will not be treated as a valid compaction anchor: `_walk_partition_chain` collects all reachable docs, but `_rebuild_from_chain` filters to `is_compacted=True` docs before validating ancestry, so an `is_compacted=False` doc is silently passed over during reconstruction.
2. **Generate the summary** by calling the LLM with the snapshot's content and a structured summarization prompt. Existing `ancestor_summaries` are included so the summary reflects the full accumulated conversation history, not only the current partition's raw turns.
3. **Update the ES document** with `is_compacted=True`, the generated summary, and the boundary metadata (`boundary_hash`, `boundary_message_count`, `tail_hash`).
4. **Atomic Redis swap** using `WATCH / MULTI / EXEC`:
   - Watch the Redis key for the conversation.
   - Read the current active partition from Redis.
   - If the current partition's `partition_uuid` differs from the snapshot's, abort: the user switched branches while compaction ran, and the summary should not be applied to a different branch.
   - Verify that the current partition's prefix still matches the snapshot (same `raw_message_start_index`, `ancestor_summaries`, and leading items). If the prefix changed, abort similarly.
   - Identify any items appended to the current partition after the snapshot was taken (new exchanges during compaction latency).
   - Construct the new active partition: `ancestor_summaries` extended with the new summary, items set to the recency tail from the snapshot plus all post-snapshot items, `raw_message_start_index` advanced to the start of the recency tail.
   - Execute the swap atomically. If `WatchError` fires (contention in the narrow read-build-write window), retry from the top of the WATCH loop.
5. **Persist the new active partition** to Elasticsearch.
6. **Release the compaction lock**.

The WATCH/MULTI/EXEC swap ensures that messages sent during the LLM summary call are never lost. Same-branch new messages accumulate in the Redis partition under the same `partition_uuid` (because the client echoes back the pre-compaction UUID until the swap completes). The compactor carries those items forward into the new partition.

---

## Branch and Retry Handling

The UI represents conversations as a message tree. Users can edit earlier messages or regenerate responses, creating branches. Compaction interacts with branching in three distinct ways:

### Retry Within the Active Partition

The most common case. `sync_from_conversation` truncates at the divergence point and appends the new message. No compaction logic is involved; the `partition_uuid` is unchanged.

### Retry Before the Recency Tail

The retry point falls within content that has been compacted away (before the recency tail boundary, i.e., before `raw_message_start_index` of the active partition). The incoming message count is smaller than `raw_message_start_index`, so `sync_from_conversation` raises `ConversationOutsideRawWindowError` immediately. The exact ES load returns `None` (the partition is `is_compacted`). The chain reconstruction then validates ancestry via `boundary_hash`; because the retry message count is less than the compacted ancestor's `boundary_message_count`, no ancestor validates. The result is a **clean partition with no ancestor summaries**, built from the incoming messages alone.

### Retry Within the Recency Tail

The retry point falls within the verbatim content retained in the active partition (between `raw_message_start_index` and the end of the recency tail). The Redis partition can be synced successfully (message count ≥ `raw_message_start_index`), so the active partition P2 is used — including its `ancestor_summaries`. The model therefore receives the compaction summary as context alongside the raw recency-tail messages. This is intentional: the recency-tail messages may reference content from the compacted region, and stripping the summary would deprive the model of the context needed to interpret them correctly.

### Branch Switch (Navigation Between Forks)

When the user navigates to a different branch via the ⟨/⟩ controls, the client sends the `partition_uuid` stored on the last node of the target branch. If that partition is in Redis (recently visited), the fast path applies. Otherwise the server cold-loads from ES or performs chain reconstruction. Only one branch is hot in Redis at a time; switching always pays the cold-load cost for the newly activated branch.

---

## Key Design Decisions

**Summaries in the system/developer message, not as conversation turns.**
Injecting ancestor summaries as a leading user/assistant exchange would conflate them with the actual conversation and could confuse attribution. Placing them in the system or developer message keeps them clearly framed as background context.

**`is_compacted=False` written before summarization.**
Writing the ES document with `is_compacted=False` before the LLM call provides a crash-recovery marker. If the server dies mid-compaction, the document exists in ES but chain walks skip it (they require `is_compacted=True`), so no stale or empty summary is ever applied. Once summarization succeeds the document is updated atomically to `is_compacted=True`.

**Synchronous `finalize()` when compaction is pending.**
The default path backgrounds `finalize()`. When compaction is about to start, `finalize()` is awaited synchronously before the background task is fired. This eliminates the race where a delayed `finalize()` SET could overwrite the result of the compaction swap and lose the newly compacted partition.

**Boundary hash over the full prefix, not just the tail.**
`tail_hash` is stored as a lookup convenience but is not used alone to prove summary applicability. `boundary_hash` covers the entire role/content sequence up to the compaction boundary. This prevents a forked branch from inheriting a summary produced by a different history that happened to share the same recent user messages.

**One Redis slot per conversation, not per partition.**
Redis holds exactly one partition per `conversation_uuid`. Switching branches always evicts the previous branch. The cost of a branch switch (one or more ES GETs) is accepted in exchange for simpler cache semantics and bounded Redis memory usage. After a cold-load, the branch is promoted to Redis and subsequent requests on it are fast.

**Compaction lock prevents duplicate concurrent compactions.**
If two requests arrive near-simultaneously and both see a high `context_pct`, only the first to acquire `compaction_lock:{conversation_uuid}` fires a compaction. The second skips compaction entirely — the in-flight compaction will produce a new partition that incorporates both requests' content (because the atomic swap carries post-snapshot items forward).

**Ancestor summaries cached in Redis alongside raw items.**
Because summaries are immutable, they can safely travel with the partition through Redis. This means the ES chain walk is paid once per branch reconstruction, not on every request.

**Exact partition load skips compacted documents.**
`_load_exact_partition` returns `None` for any document with `is_compacted=True`. This prevents the server from resuming a sealed partition directly and forces it to go through chain reconstruction, which correctly assembles summaries and sets `raw_message_start_index`.

**Raw items preserved in ES indefinitely.**
`items_json` is stored in ES for every partition, compacted or not. This enables future "deep rehydration" — deserializing the full original items instead of relying on the summary when branching into a compacted region, at the cost of a larger context window.

---

## Known Limitations and Deferred Work

**Deep rehydration.** When a retry lands before the recency tail, the server currently builds a fresh partition from the incoming messages with no ancestor summaries. The full original context for that branch is available in `items_json` in ES but is not yet used. A future deep-rehydration mode could restore full fidelity at higher token cost.

**Script and eval harnesses.** `ScriptHarness` and `EvalHarness` use `ContextPartition` without Redis or Elasticsearch. Compaction is not implemented for those harnesses. Long script or eval sessions that approach the context limit will fail. Simplified in-memory summarization at threshold could be added later.

**Semantic search over history.** The `message_content` and `summary` fields in Elasticsearch are full-text indexed in preparation for conversational history search. `ContextPartitionSearcher.search_partitions` exists as the hook for this but is not yet wired into any harness endpoint.

---

## Relevant Code Files

| File | Role |
|---|---|
| `prokaryotes/api_v1/models.py` | `ContextPartition` (with new compaction fields), `ChatConversation`, `ContextPartitionItem`, `compute_boundary_hash`, `compute_tail_hash`, `conversation_message_items`, exception classes |
| `prokaryotes/web_v1/__init__.py` | `WebBase.sync_context_partition`, `_compact_partition`, `_rebuild_from_chain`, `_boundary_message_items_for_partition`, `stream_and_finalize`, `finalize`, `_load_exact_partition`, `_walk_partition_chain`, `_recency_tail_items`, `get_compaction_status` |
| `prokaryotes/search_v1/context_partitions.py` | `ContextPartitionSearcher` mixin: `get_partition`, `put_partition`, `update_partition`, `find_partition_by_tail_hash`, `search_partitions`; Elasticsearch index mapping |
| `prokaryotes/search_v1/__init__.py` | `SearchClient` inheriting `ContextPartitionSearcher` |
| `prokaryotes/anthropic_v1/web_harness.py` | `on_usage` closure, `pending_compaction` flag, `compact` closure, `_summarize_and_compact` |
| `prokaryotes/anthropic_v1/__init__.py` | `context_pct` event emission after each LLM round |
| `prokaryotes/openai_v1/web_harness.py` | Same as Anthropic harness; developer message assembly including `ancestor_summaries` |
| `prokaryotes/openai_v1/__init__.py` | `context_pct` event emission |
| `prokaryotes/utils_v1/llm_utils.py` | `MODEL_CONTEXT_WINDOWS`, `COMPACTION_TOKEN_THRESHOLD_PCT`, `COMPACTION_RECENCY_TAIL`, `COMPACTION_LOCK_TTL_SECONDS` |
| `scripts/search_init.py` | Creates or updates the `context-partitions` Elasticsearch index |
| `scripts/static/ui.js` | `getLastPartitionUuid`, `partition_uuid` event handling, `context_pct` fill indicator, compaction indicator show/clear logic, `startCompactionPolling`/`stopCompactionPolling` (5 s poll against `/compaction-status`) |
| `tests/context_partition_utils.py` | Shared test infrastructure: `FakeRedis`, `FakeSearchClient`, `make_doc`, `make_web_base`, `make_message_items`, `make_chat_messages` |
| `tests/test_compaction_swap.py` | `_compact_partition` internals: post-snapshot carry-forward, `raw_message_start_index` advancement, UUID/prefix/missing-Redis abort guards, empty-summary and exception lock cleanup, multi-generation summary accumulation, WatchError retry, full-prefix boundary-hash storage |
| `tests/test_compaction_rebuild.py` | `sync_context_partition` compaction edge cases (retry-before-raw-window, edit-within-tail), `stream_and_finalize` (compaction events, duplicate-lock guard), `_walk_partition_chain` integrity (missing intermediate, UUID mismatch, cycle), `_rebuild_from_chain` single- and multi-generation reconstruction |
| `tests/test_compaction_status.py` | `get_compaction_status` endpoint: lock-present returns not-done, partition unchanged returns not-done, partition changed returns done, partition evicted from Redis returns done |
| `tests/test_compaction_provider.py` | `to_anthropic_messages` with ancestor summaries; Anthropic and OpenAI `_summarize_and_compact` ancestor-summary injection; OpenAI `text_preamble` exclusion from compaction payload |
| `tests/test_api_v1_models.py` | Unit tests for hash functions and `ContextPartition` sync (progress, retry, edit paths) |
| `tests/test_web_v1.py` | `_recency_tail_items`, `finalize`, basic `sync_context_partition` Redis-path tests |
| `tests/test_search_v1_context_partitions.py` | Tests for `ContextPartitionSearcher` methods |
| `tests/test_anthropic_v1.py` | Tests for `context_pct` emission |
| `tests/test_openai_v1.py` | Tests for `context_pct` emission |
