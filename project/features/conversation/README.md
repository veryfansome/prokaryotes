# Conversation Model

## Overview

A `Conversation` is the single primitive for an external dialogue — one web chat session today, one Slack thread in the future. Single-user is the degenerate case of multi-user where the author cardinality is 1.

Three properties define the model:

1. **Source-ID-based reconciliation.** Each message carries a stable, server-assigned `source_id`. Reconciliation diffs an incoming message list against the stored snapshot by `source_id` and classifies the change (append, edit, delete, divergence). Position is irrelevant.
2. **External dialogue and turn execution are separate.** A `Conversation` is the append-only log of external messages (what users posted, what the bot posted). A `TurnExecution` is the LLM-internal log for one bot reply, holding tool-call records only. Intermediate narration the model produces between tool calls is transient and never persisted.
3. **Multi-author by default.** `ConversationMessage` carries `author_id`; the `user` vs `assistant` role is derived at projection time by comparing to the conversation's `bot_author_id`.

[Compaction](../compaction/README.md) — periodic summarization of old history so a conversation can run past the context window — is a lifecycle operation layered on this model; that doc owns the compaction-specific machinery.

---

## Core Concepts

### `Conversation`

A persistent snapshot of an external dialogue, identified by a stable `conversation_uuid`. A single `conversation_uuid` is a *DAG* of snapshots, each with its own `snapshot_uuid` — linear children via compaction, sibling branches via edit/regenerate (see [Branches and Snapshots](#branches-and-snapshots)).

A snapshot owns its ordered `ConversationMessage`s, the bot's `author_id`, the snapshot-DAG metadata (`parent_snapshot_uuid`, `ancestor_summaries`, `raw_message_start_index`), and the tool state lifted across compaction (`lifted_turn_items` + `lifted_anchor_source_id`).

### `ConversationMessage`

One external dialogue message: `source_id` (identity *and* ordering key), `author_id` (opaque; bot vs human is derived against `Conversation.bot_author_id`), `content`, `display_name`, and the `deleted` / `edited` storage flags.

`source_id` is surface-assigned, sortable, and formatted `seconds.microseconds` so lexicographic order equals chronological order. The web syncer assigns one on first encounter; Slack supplies the message `ts`. Monotonicity within a conversation is enforced — a candidate `<=` the last assigned `source_id` is bumped by one microsecond.

### `TurnExecution` and `TurnItem`

`TurnExecution` is the LLM-internal log for one bot reply, stored in its own ES index keyed by the resulting bot message's `source_id`. It holds only `TurnItem`s — `function_call` / `function_call_output` records. The final assistant message itself lives on the `Conversation` (so external surfaces can see and edit it), not in the `TurnExecution`.

**Transient-narration invariant.** Intermediate assistant text between tool calls ("let me check…") is *not* persisted. The model writes it as in-the-moment scratch work; replaying it on a later turn would condition the model on past-tense versions of itself. The LLM client keeps it in a streaming buffer during the active turn and discards it at finalization — only `function_call` / `function_call_output` items are committed. `tests/unit_tests/test_anthropic_v1.py` and `test_openai_v1.py` guard this.

---

## Data Model

### Pydantic models — `prokaryotes/conversation_v1/models.py`

| Model | Role |
|---|---|
| `Conversation` | A snapshot: `conversation_uuid`, `snapshot_uuid`, `parent_snapshot_uuid`, `bot_author_id`, `ancestor_summaries`, `lifted_turn_items`, `lifted_anchor_source_id`, `raw_message_start_index`, `messages`. |
| `ConversationMessage` | External message: `source_id`, `author_id`, `content`, `display_name`, `deleted`, `edited`. |
| `TurnExecution` | Per-bot-reply tool log: `conversation_uuid`, `bot_message_source_id`, `items`, `completed`. |
| `TurnItem` | One `function_call` / `function_call_output` record. Carries `prokaryotes_annotations` (file-tool annotations live here). |
| `ProjectedItem` | LLM-bound projection — the bridge from storage to provider wire format. |
| `NormalizedMessage` | What `reconcile()` operates on — fully-populated incoming message. |

Wire-format models live in `prokaryotes/api_v1/models.py`: `IncomingMessage` / `IncomingConversation` (the `POST /chat` body) and `CompactionStatusResponse`.

### Elasticsearch

Two indices, both `dynamic: strict` — `prokaryotes/search_v1/conversations.py`:

- **`conversations`** — one document per snapshot. Carries the model fields plus `compaction_state`, `is_compacted`, `summary`, `boundary_hash` / `tail_hash`, and a full-text-indexed `message_content`. `messages_json` is retained even after `is_compacted=True` so the [assistant-message guardrail](#assistant-message-guardrail) can read compacted ancestors' per-message identity.
- **`turn-executions`** — one document per bot reply that involved tool calls, keyed `conversation_uuid:bot_message_source_id`.

`scripts/search_init.py` bootstraps both indices (plus `topics`).

### Redis

| Key | Purpose |
|---|---|
| `conversation:{conversation_uuid}` | Cached active `Conversation` snapshot (one per conversation). |
| `assistant_index:{conversation_uuid}` | `{bot_source_id: content_hash}` for the assistant-message guardrail. |
| `compaction_lock:{conversation_uuid}` | Compaction CAS lock. |
| `compaction_status:{pending_snapshot_uuid}` | `/compaction-status` relabel target. |

TTLs default to `CONVERSATION_CACHE_EXPIRY_SECONDS` (7 days).

---

## Reconciliation

`reconcile(stored, incoming)` in `prokaryotes/conversation_v1/reconcile.py` diffs incoming `NormalizedMessage`s against `stored.messages` by `source_id` and returns a classification plus an operation list — it does not apply anything:

- **`match`** — no operations.
- **`append`** — every stored non-deleted `source_id` is in incoming with matching content, and incoming has additional `source_id`s at the end.
- **`edit`** — every stored `source_id` is in incoming, but at least one content differs.
- **`delete`** — stored `source_id`s missing from incoming, with no fresh `source_id`s and no content changes.
- **`divergence`** — anything else (e.g. an edit/delete that isn't a pure trailing change).

The **syncer** decides how to apply per surface. The web syncer (`ConversationSyncer._apply_result`) mutates in place for `match` / `append` and branches a new snapshot via `_apply_divergence` for `edit` / `delete` / `divergence` — this preserves the snapshot-DAG branch contract (the original snapshot stays intact in ES). `SlackConversationSyncerMixin` overrides this to mutate in place for all classifications, since Slack threads are linear and authoritative; its delete path also re-keys a tombstoned bot's `TurnExecution` to the next non-tombstoned bot in the run.

### Three-tier load — `ConversationSyncer.sync_conversation`

`prokaryotes/context_v1/conversation_sync.py` reconciles the active snapshot per request:

1. **Redis fast path.** The cache "follows" the client when its `snapshot_uuid` equals the client's *or its parent* (a post-compaction relabel the client hasn't applied yet).
2. **Exact ES load.** Fetch the doc at the client's `snapshot_uuid` by `_id`. A compacted doc is forwarded to chain rebuild — a compacted snapshot can't be the active one directly.
3. **Ancestor-chain rebuild.** Walk `parent_snapshot_uuid` links; validate a compacted ancestor's `boundary_hash` against a prefix of incoming. No matching ancestor → a fresh root `Conversation`.

After load, the post-load pipeline runs: `_split_compacted_prefix` → `_detect_unacknowledged_bot_messages` → `_assign_source_ids` → `reconcile` → `_apply_result` → `_cache_and_persist_conversation`.

---

## Projection

`project_for_llm` (`prokaryotes/conversation_v1/project.py`) is the single bridge from storage to the LLM API and the only place role assignment, display-name prefixing, and consecutive-same-role text merging happen. Everything else in the system is role-agnostic.

- Role is derived: `author_id == bot_author_id` → `assistant`, else `user`.
- In a multi-author conversation (more than one distinct human `author_id`), each human turn is prefixed `<display_name> `.
- Each bot message's `TurnExecution.items` are interleaved immediately before it; `lifted_turn_items` emit just before the `lifted_anchor_source_id` bot's tool round.
- Consecutive same-role `message` items are merged (`\n\n`-joined) — this is what keeps the OpenAI Responses API's alternation requirement satisfied.

LLM clients translate `list[ProjectedItem]` to their wire format (`_items_to_anthropic_messages` / `_items_to_openai_input`). The instruction (system/developer) message is injected by the harness at position 0 *after* projection — it is not part of conversation storage.

`current_turn_items(conversation, historical_turns, active_turn)` exposes the flat `TurnItem` view tools with lift semantics (`FileTool`) read from, so they see live windows carried across compaction.

---

## Branches and Snapshots

A `conversation_uuid` is a DAG of snapshots. Two edge kinds, both via `parent_snapshot_uuid`, distinguished by the parent's state:

- **Compaction edge** (linear) — the parent is `is_compacted=true`; its raw messages were summarized into the child's `ancestor_summaries`.
- **Branch edge** (sibling) — the parent is non-compacted; the child holds a tail that diverged from the parent's. Web edit/regenerate produces these.

### Divergence creates a new snapshot

When the web syncer applies `edit` / `delete` / `divergence`, `_apply_divergence` roots a new snapshot at the shared prefix. The original snapshot is untouched in ES — prior branches survive server-side.

- **Case A — divergence within the parent's raw window** (the common edit/regenerate path). The child inherits the parent's `ancestor_summaries` and `raw_message_start_index` verbatim, and recomputes `lifted_turn_items` / `lifted_anchor_source_id` against its own raw window.
- **Case B — divergence before the raw window** (editing a message that was compacted away). The compacted-prefix split fails, the syncer discards the loaded snapshot, and a fresh root `Conversation` is built from incoming alone — no inherited summaries, `raw_message_start_index=0`.

Child branch `messages` are sorted by `source_id`, so the stored list stays in chronological order regardless of request shape.

### Client-side branch model — `scripts/static/`

The browser keeps a `messageTree`. Edit and regenerate create *new* sibling nodes; each node stores its server-assigned `source_id` and the `snapshot_uuid` of the snapshot it belongs to. `conversation_client.js` holds the pure protocol primitives (`applyHandshake`, `applyBotMessage`, `relabelSnapshotUuid`, `applyResyncHandshake`, `buildRequestMessages`); `ui.js` wires them into the chat UI.

Because branches share a prefix, a shared node belongs to several snapshots and `applyHandshake` restamps it each turn. The request's `snapshot_uuid` is therefore anchored on the **leaf of the branch the user is viewing**, captured before any tree mutation — not on the edit/regenerate parent, which is a shared node a sibling branch may have restamped.

Cross-session branch survival is out of scope: snapshots persist in ES, but the client's per-node `snapshot_uuid` mapping dies with the page session, so a reload resumes only the most recently active branch.

---

## Web Wire Protocol

### Request — `POST /chat`

`IncomingConversation`: `{conversation_uuid, snapshot_uuid?, messages: [{role, content, source_id?}]}`. `source_id` is omitted for newly-authored messages; the syncer assigns one. The route normalizes each message into a `NormalizedMessage` using session info (`author_id` from `chat_user.id`, `display_name` from `full_name`).

### Response stream (NDJSON) — event ordering is contractual

- **Handshake** — always first. `{snapshot_uuid, source_id_assignments: [{client_index, source_id}]}`. The client stamps both `source_id` and `snapshot_uuid` onto every submitted user node. Invariant: every node with a `source_id` also has a `snapshot_uuid`.
- `context_pct`, `text_delta`, `tool_call`, `progress_message` flow during the turn.
- **`bot_message`** — `{bot_message: {source_id}}`, emitted exactly once after the final assistant message commits. The client creates the assistant node only on this event; a stream that aborts before it creates no assistant node.
- **`compaction_pending`** — optional last event; see [compaction](../compaction/README.md).

### Stream-loss recovery

- **Pre-commit.** If the stream aborts after the handshake but before `bot_message`, the user node already carries the new `snapshot_uuid`, so a retry extends the branch instead of re-diverging.
- **Post-commit (resync).** If the server committed a bot message the client never saw, the next POST looks like a fork. The server detects the trailing-bot-skip pattern *before* reconcile, emits a resync handshake with `unacknowledged_bot_messages`, and closes the stream without an LLM call. The client reconstructs the missing assistant nodes; send-from-leaf auto-retries, edit/regenerate pops the pending message back to a draft.

### Assistant-message guardrail

Web clients only ever author `user` messages. `validate_assistant_messages` runs before `sync_conversation` and rejects (4xx) any `role="assistant"` entry whose `source_id` is unknown to the conversation DAG, or whose content hash doesn't match storage. Recognition is DAG-scoped — it reads compacted ancestors' retained `messages_json` — so the Case B retry path, which legitimately echoes compacted ancestors' assistant `source_id`s, is accepted.

---

## Surface Mappings

| Concept | Web | Slack (future) |
|---|---|---|
| `conversation_uuid` | Server-generated on first message | `uuid5(SLACK_NS, team:channel:thread)` |
| `bot_author_id` | Constant `"__bot__"` | `bot_user_id` |
| `source_id` | Syncer-assigned `seconds.microseconds` | Slack `ts` |
| `author_id` | `chat_user.id` | Slack `user` |
| Apply policy | Branch on divergence | Overwrite in place |

`SlackConversationSyncerMixin` and the tombstone re-keying machinery exist; no concrete Slack harness ships yet.

---

## Out of Scope

- Cross-surface conversation merging (web + Slack sharing one history).
- Per-message rich content (`content` is intentionally `str`).
- Cross-session branch survival (the ES DAG could support it later).
- Branch-prefix deduplication (sibling snapshots duplicate the shared prefix).
- `TurnExecution` garbage collection (orphaned turns linger in ES).

---

## Relevant Code Files

| File | Role |
|---|---|
| `prokaryotes/conversation_v1/models.py` | `Conversation`, `ConversationMessage`, `TurnExecution`, `TurnItem`, `ProjectedItem`, `NormalizedMessage`, `compute_boundary_hash` / `compute_tail_hash`. |
| `prokaryotes/conversation_v1/reconcile.py` | Source-ID-keyed `reconcile()`. |
| `prokaryotes/conversation_v1/project.py` | `project_for_llm`, `current_turn_items`. |
| `prokaryotes/context_v1/conversation_sync.py` | `ConversationSyncer` (three-tier load, post-load pipeline, `_apply_divergence`, guardrail), `SlackConversationSyncerMixin`. |
| `prokaryotes/api_v1/models.py` | `IncomingMessage` / `IncomingConversation`, `CompactionStatusResponse`. |
| `prokaryotes/search_v1/conversations.py` | `ConversationSearcher`: ES CRUD + index mappings. |
| `prokaryotes/harness_v1/base.py` / `web.py` | `stream_and_finalize`, `finalize_turn`; the `/chat` route, projection, instruction assembly. |
| `prokaryotes/anthropic_v1/`, `openai_v1/` | `ProjectedItem` → provider wire format; the transient-narration buffer. |
| `scripts/static/conversation_client.js`, `ui.js` | Client-side branch model, wire protocol, branch-anchored `snapshot_uuid` selection. |
| `tests/unit_tests/test_reconcile.py`, `test_conversation_*.py`, `test_project_for_llm.py` | Reconcile matrix, syncer, projection. |
| `tests/integration_tests/tier_b/test_unified_web_flow.py`, `test_unified_multi_author_flow.py` | Edit/regenerate branching and multi-author projection end to end. |
